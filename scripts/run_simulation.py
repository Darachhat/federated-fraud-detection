"""
Single-Process Simulation Mode.
Runs the complete federated learning protocol —
Global Server + all three client banks — in a
single terminal process for debugging and rapid
experiment iteration.

No network communication or multi-threading required.
All federation rounds execute sequentially in memory.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --rounds 3
    python scripts/run_simulation.py --rounds 5 --seed 42
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.preprocessor import PaySimPreprocessor
from src.evaluation.metrics import evaluate_model
from src.federated.aggregator import JSONTreeConcatenator
from src.models.xgboost_trainer import XGBoostTrainer
from src.utils.io_utils import (
    load_config, load_dataframe,
    save_results
)
from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed

BASE_CONFIG  = "config/base_config.yaml"
BANK_CONFIGS = [
    "config/bank1_config.yaml",
    "config/bank2_config.yaml",
    "config/bank3_config.yaml",
]


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-process FL simulation mode."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of federation rounds (overrides config)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────

class FederatedSimulation:
    """
    Self-contained federated learning simulation.

    Executes the full FL protocol in a single process:
        1. Load datasets for all three banks
        2. Run Local-Only baseline (Round 0)
        3. Execute N federation rounds:
           a. Each bank trains locally
           b. Global Server applies JSON Tree Concatenation
           c. Federated model redistributed to all banks
           d. All banks evaluated on global test set
        4. Save complete results trajectory
    """

    TARGET = "isFraud"

    def __init__(
        self,
        base_config: Dict,
        bank_configs: List[Dict],
        num_rounds: Optional[int] = None
    ):
        self.base_config  = base_config
        self.bank_configs = bank_configs
        self.num_rounds   = (
            num_rounds
            or base_config["federated"]["num_rounds"]
        )
        self.threshold    = base_config["evaluation"]["threshold"]
        self.results_dir  = base_config["results"]["output_dir"]
        self.exchange_dir = base_config["federated"]["exchange_dir"]
        self.global_dir   = base_config["federated"]["global_model_dir"]

        self.logger = setup_logger(
            "simulation",
            log_dir=base_config["logging"]["log_dir"]
        )
        self.aggregator = JSONTreeConcatenator()

        # Create required directories
        for d in [self.exchange_dir, self.global_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Per-bank state
        self.trainers:       Dict[str, XGBoostTrainer]  = {}
        self.train_dfs:      Dict[str, pd.DataFrame]    = {}
        self.global_test_df: Optional[pd.DataFrame]     = None

        # Results accumulator
        # Structure: {bank_id: [round_0_results, round_1_results, ...]}
        self.trajectory: Dict[str, List[Dict]] = {
            cfg["client"]["bank_id"]: []
            for cfg in bank_configs
        }

    # ── Data Loading ─────────────────────────────────────────

    def load_data(self) -> None:
        """Load all bank training sets and the global test set."""
        self.logger.info("Loading datasets...")

        # Global test set — loaded once, shared across all banks
        global_test_path = self.bank_configs[0]["data"]["global_test_path"]
        self.global_test_df = load_dataframe(global_test_path)

        self.logger.info(
            f"Global test set: {len(self.global_test_df):,} records "
            f"| Fraud: {int(self.global_test_df[self.TARGET].sum()):,}"
        )

        # Per-bank training sets
        for cfg in self.bank_configs:
            bank_id   = cfg["client"]["bank_id"]
            train_df  = load_dataframe(cfg["data"]["train_path"])
            self.train_dfs[bank_id] = train_df
            self.trainers[bank_id]  = XGBoostTrainer(self.base_config)

            fraud_count = int(train_df[self.TARGET].sum())
            fraud_pct   = train_df[self.TARGET].mean() * 100

            self.logger.info(
                f"  {bank_id}: {len(train_df):,} records "
                f"| Fraud: {fraud_count:,} ({fraud_pct:.4f}%)"
            )

    # ── Round 0: Local-Only Baseline ─────────────────────────

    def run_local_baseline(self) -> None:
        """
        Train and evaluate isolated local models for all banks.
        Establishes Round 0 performance — the blind spot baseline.
        """
        self._print_round_header(0, label="LOCAL-ONLY BASELINE")

        for cfg in self.bank_configs:
            bank_id  = cfg["client"]["bank_id"]
            trainer  = self.trainers[bank_id]
            train_df = self.train_dfs[bank_id]

            self.logger.info(
                f"[{bank_id.upper()}] Training local-only model..."
            )

            # Train from scratch — no federation
            trainer.train(train_df=train_df)

            # Save local model
            model_path = str(
                Path(self.base_config["federated"]["local_model_dir"])
                / f"{bank_id}_local_model.json"
            )
            trainer.save(model_path)

            # Evaluate on global test set
            results = evaluate_model(
                model=trainer.model,
                test_df=self.global_test_df,
                threshold=self.threshold,
                bank_id=bank_id,
                round_num=0
            )
            self.trajectory[bank_id].append(results)

        # Save Round 0 results
        self._save_round_results(round_num=0)

    # ── Federation Rounds 1–N ────────────────────────────────

    def run_federation_rounds(self) -> None:
        """Execute all N federated communication rounds."""

        # Track current global model path across rounds
        current_global_path: Optional[str] = None

        for round_num in range(1, self.num_rounds + 1):
            self._print_round_header(round_num)

            # ── Phase A: Local Training (all banks) ──────────
            round_model_paths = []

            for cfg in self.bank_configs:
                bank_id  = cfg["client"]["bank_id"]
                trainer  = self.trainers[bank_id]
                train_df = self.train_dfs[bank_id]

                self.logger.info(
                    f"[{bank_id.upper()}] "
                    f"Round {round_num} local training..."
                )

                # Warm-start from previous global model
                trainer.train(
                    train_df=train_df,
                    init_model_path=current_global_path
                )

                # Serialize to exchange directory
                exchange_path = str(
                    Path(self.exchange_dir)
                    / f"{bank_id}_round_{round_num}.json"
                )
                trainer.save(exchange_path)
                round_model_paths.append(exchange_path)

            # ── Phase B: Server Aggregation ───────────────────
            self.logger.info(
                f"\n[SERVER] Round {round_num} — "
                f"Applying JSON Tree Concatenation Algorithm..."
            )

            global_model_path = str(
                Path(self.global_dir)
                / f"global_model_round_{round_num}.json"
            )

            self.aggregator.aggregate(
                client_model_paths=round_model_paths,
                base_model_path=round_model_paths[0],
                output_path=global_model_path
            )

            current_global_path = global_model_path

            # ── Phase C: Load Federated Model + Evaluate ──────
            for cfg in self.bank_configs:
                bank_id = cfg["client"]["bank_id"]
                trainer = self.trainers[bank_id]

                # Load the federated global model
                trainer.load(global_model_path)

                # Evaluate on global test set
                results = evaluate_model(
                    model=trainer.model,
                    test_df=self.global_test_df,
                    threshold=self.threshold,
                    bank_id=bank_id,
                    round_num=round_num
                )
                self.trajectory[bank_id].append(results)

            # ── Phase D: Clean exchange directory ─────────────
            for path in round_model_paths:
                Path(path).unlink(missing_ok=True)

            # Save round results
            self._save_round_results(round_num=round_num)

    # ── Results Persistence ───────────────────────────────────

    def _save_round_results(self, round_num: int) -> None:
        """Save all bank results for a specific round."""
        round_data = {}
        for bank_id, results_list in self.trajectory.items():
            if results_list:
                round_data[bank_id] = results_list[-1]

        save_results(
            round_data,
            output_dir=str(
                Path(self.results_dir)
                / "federated"
                / f"round_{round_num}"
            ),
            filename=f"all_banks_round_{round_num}"
        )

    def save_full_trajectory(self) -> None:
        """Save the complete per-bank per-round trajectory."""
        for bank_id, results_list in self.trajectory.items():
            save_results(
                {"trajectory": results_list},
                output_dir=str(
                    Path(self.results_dir) / "federated"
                ),
                filename=f"{bank_id}_full_trajectory"
            )

        # Save consolidated trajectory for all banks
        save_results(
            self.trajectory,
            output_dir=str(
                Path(self.results_dir) / "federated"
            ),
            filename="all_banks_full_trajectory"
        )
        self.logger.info("Full trajectory saved. ✓")

    # ── Utilities ─────────────────────────────────────────────

    def _print_round_header(
        self,
        round_num: int,
        label: Optional[str] = None
    ) -> None:
        title = label or f"FEDERATION ROUND {round_num} / {self.num_rounds}"
        print(
            f"\n{'═' * 60}\n"
            f"  {title}\n"
            f"{'═' * 60}"
        )

    def print_summary(self) -> None:
        """Print a final performance summary table."""
        print(
            f"\n{'═' * 70}\n"
            f"  SIMULATION COMPLETE — PERFORMANCE SUMMARY\n"
            f"{'═' * 70}"
        )

        header = (
            f"  {'Bank':<12} "
            f"{'Round':<8} "
            f"{'AUPRC':<10} "
            f"{'F1-Score':<10} "
            f"{'Precision':<12} "
            f"{'Recall':<10}"
        )
        print(header)
        print("  " + "─" * 60)

        for bank_id, results_list in self.trajectory.items():
            for r in results_list:
                print(
                    f"  {bank_id:<12} "
                    f"{r['round']:<8} "
                    f"{r['auprc']:<10.4f} "
                    f"{r['f1_score']:<10.4f} "
                    f"{r['precision']:<12.4f} "
                    f"{r['recall']:<10.4f}"
                )
            print("  " + "─" * 60)

        print(
            f"\n  [NOTE] Accuracy excluded from all results.\n"
            f"         It is a misleading metric under 0.13% "
            f"fraud prevalence.\n"
        )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    args       = parse_args()
    base_cfg   = load_config(BASE_CONFIG)
    bank_cfgs  = [load_config(p) for p in BANK_CONFIGS]

    # Override config values from CLI arguments
    seed = args.seed or base_cfg["project"]["seed"]
    set_global_seed(seed)

    print(
        f"\n{'═' * 60}\n"
        f"  FEDERATED LEARNING SIMULATION\n"
        f"  Single-process mode — all banks + server\n"
        f"  Seed: {seed}\n"
        f"{'═' * 60}\n"
    )

    sim = FederatedSimulation(
        base_config=base_cfg,
        bank_configs=bank_cfgs,
        num_rounds=args.rounds
    )

    start_time = time.time()

    # ── Step 1: Load all data ──
    sim.load_data()

    # ── Step 2: Local-Only baseline (Round 0) ──
    sim.run_local_baseline()

    # ── Step 3: Federation rounds 1–N ──
    sim.run_federation_rounds()

    # ── Step 4: Save complete trajectory ──
    sim.save_full_trajectory()

    # ── Step 5: Print summary table ──
    sim.print_summary()

    elapsed = time.time() - start_time
    print(
        f"  Total simulation time: "
        f"{elapsed:.1f}s ({elapsed / 60:.1f} min)\n"
    )


if __name__ == "__main__":
    main()