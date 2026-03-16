"""
Centralized Baseline Experiment.
Trains a single XGBoost model on the fully pooled
dataset across all three banks — privacy-violated
upper bound for performance comparison.

Usage:
    python scripts/run_baseline_central.py
"""

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.models.xgboost_trainer import XGBoostTrainer
from src.utils.io_utils import load_config, load_dataframe, save_results
from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed

BASE_CONFIG = "config/base_config.yaml"
BANK_CONFIGS = [
    "config/bank1_config.yaml",
    "config/bank2_config.yaml",
    "config/bank3_config.yaml",
]


def main():
    base_config = load_config(BASE_CONFIG)
    set_global_seed(base_config["project"]["seed"])

    logger = setup_logger(
        "central_baseline",
        log_dir=base_config["logging"]["log_dir"]
    )

    print("\n" + "═" * 60)
    print("  CENTRALIZED BASELINE EXPERIMENT")
    print("  WARNING: This condition pools raw data across all")
    print("  banks — it is PRIVACY-VIOLATING by design and")
    print("  serves only as a theoretical upper bound.")
    print("═" * 60 + "\n")

    # ── Step 1: Pool training data across all three banks ──
    train_frames = []

    for bank_cfg_path in BANK_CONFIGS:
        bank_config = load_config(bank_cfg_path)
        bank_id = bank_config["client"]["bank_id"]
        train_path = bank_config["data"]["train_path"]

        df = load_dataframe(train_path)
        train_frames.append(df)

        logger.info(
            f"Loaded {bank_id}: {len(df):,} records "
            f"| Fraud: {int(df['isFraud'].sum()):,}"
        )

    pooled_train_df = pd.concat(train_frames, ignore_index=True)

    logger.info(
        f"\nPooled training set: {len(pooled_train_df):,} records "
        f"| Total fraud: {int(pooled_train_df['isFraud'].sum()):,} "
        f"| Fraud prevalence: "
        f"{pooled_train_df['isFraud'].mean() * 100:.4f}%"
    )

    # ── Step 2: Load global test set ──
    # Use Bank 1's config to get global test path
    # (same path referenced in all bank configs)
    bank1_config = load_config(BANK_CONFIGS[0])
    global_test_df = load_dataframe(
        bank1_config["data"]["global_test_path"]
    )

    logger.info(
        f"Global test set: {len(global_test_df):,} records "
        f"| Fraud: {int(global_test_df['isFraud'].sum()):,}"
    )

    # ── Step 3: Train centralized XGBoost model ──
    logger.info("Training centralized XGBoost model on pooled data...")

    trainer = XGBoostTrainer(base_config)
    trainer.train(train_df=pooled_train_df)

    # ── Step 4: Save centralized model ──
    model_path = str(
        Path(base_config["federated"]["local_model_dir"])
        / "centralized_model.json"
    )
    trainer.save(model_path)
    logger.info(f"Centralized model saved → {model_path}")

    # ── Step 5: Evaluate on global test set ──
    results = evaluate_model(
        model=trainer.model,
        test_df=global_test_df,
        threshold=base_config["evaluation"]["threshold"],
        bank_id="centralized",
        round_num=0
    )

    results["note"] = (
        "Privacy-violated upper bound. "
        "Raw data pooled across all three banks. "
        "Not for production use."
    )

    # ── Step 6: Save results ──
    save_results(
        results,
        output_dir=str(
            Path(base_config["results"]["output_dir"])
            / "central_baseline"
        ),
        filename="centralized_baseline_results"
    )

    print("\n[CENTRAL] Centralized baseline experiment complete. ✓\n")
    print(
        f"  AUPRC    : {results['auprc']:.4f}\n"
        f"  F1-Score : {results['f1_score']:.4f}\n"
        f"  [Accuracy excluded — misleading under 0.13% imbalance]\n"
    )


if __name__ == "__main__":
    main()