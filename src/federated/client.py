"""
Federated Learning Client.
Handles local training, model submission,
and evaluation across all federation rounds.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import evaluate_model
from src.models.xgboost_trainer import XGBoostTrainer
from src.utils.io_utils import (
    load_config, load_dataframe,
    save_results, wait_for_file
)
from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed


class FederatedClient:
    """
    Federated Learning client representing a single bank.

    Responsibilities:
        - Train local XGBoost model each round
        - Submit serialized model to exchange directory
        - Load and apply federated global model
        - Evaluate performance at each round
        - Log all results per round
    """

    TARGET = "isFraud"

    def __init__(
        self,
        bank_config_path: str,
        base_config_path: str = "config/base_config.yaml"
    ):
        self.base_config = load_config(base_config_path)
        self.bank_config = load_config(bank_config_path)
        self.bank_id = self.bank_config["client"]["bank_id"]

        self.logger = setup_logger(
            self.bank_id,
            log_dir=self.base_config["logging"]["log_dir"]
        )

        self.trainer = XGBoostTrainer(self.base_config)
        self.exchange_dir = self.base_config["federated"]["exchange_dir"]
        self.global_dir = self.base_config["federated"]["global_model_dir"]
        self.num_rounds = self.base_config["federated"]["num_rounds"]
        self.timeout = self.base_config["federated"]["round_timeout_seconds"]
        self.results_dir = self.base_config["results"]["output_dir"]
        self.threshold = self.base_config["evaluation"]["threshold"]

        set_global_seed(self.base_config["project"]["seed"])

        # Load datasets
        self.train_df = load_dataframe(
            self.bank_config["data"]["train_path"]
        )
        self.global_test_df = load_dataframe(
            self.bank_config["data"]["global_test_path"]
        )

        self.round_results = []

    def _exchange_path(self, round_num: int) -> str:
        """Path for submitting local model to server."""
        return str(
            Path(self.exchange_dir)
            / f"{self.bank_id}_round_{round_num}.json"
        )

    def _global_model_path(self, round_num: int) -> str:
        """Path to fetch federated global model from server."""
        return str(
            Path(self.global_dir)
            / f"global_model_round_{round_num}.json"
        )

    def run(self) -> None:
        """
        Execute the full federated training protocol
        for this client across all configured rounds.
        """
        self.logger.info(
            f"Client {self.bank_id.upper()} started. "
            f"Rounds: {self.num_rounds}"
        )

        current_model_path: Optional[str] = None

        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(
                f"[{self.bank_id.upper()}] "
                f"Round {round_num} / {self.num_rounds} — "
                f"Local training..."
            )

            # Step 1 — Train local model
            # Warm-start from previous global model if available
            self.trainer.train(
                train_df=self.train_df,
                init_model_path=current_model_path
            )

            # Step 2 — Submit local model to exchange directory
            submission_path = self._exchange_path(round_num)
            self.trainer.save(submission_path)
            self.logger.info(
                f"[{self.bank_id.upper()}] "
                f"Round {round_num} model submitted ✓"
            )

            # Step 3 — Wait for federated global model from server
            global_path = self._global_model_path(round_num)
            self.logger.info(
                f"[{self.bank_id.upper()}] "
                f"Waiting for global model Round {round_num}..."
            )
            arrived = wait_for_file(global_path, self.timeout)
            if not arrived:
                raise TimeoutError(
                    f"Timeout waiting for global model "
                    f"at round {round_num}."
                )

            # Step 4 — Load federated global model
            self.trainer.load(global_path)
            current_model_path = global_path

            # Step 5 — Evaluate federated model on global test set
            results = evaluate_model(
                model=self.trainer.model,
                test_df=self.global_test_df,
                threshold=self.threshold,
                bank_id=self.bank_id,
                round_num=round_num
            )
            self.round_results.append(results)

            # Save per-round results
            save_results(
                results,
                output_dir=str(
                    Path(self.results_dir)
                    / "federated"
                    / f"round_{round_num}"
                ),
                filename=f"{self.bank_id}_round_{round_num}_results"
            )

        self.logger.info(
            f"[{self.bank_id.upper()}] "
            f"All {self.num_rounds} rounds complete."
        )
        self._save_full_trajectory()

    def _save_full_trajectory(self) -> None:
        """Save the complete per-round performance trajectory."""
        save_results(
            {"trajectory": self.round_results},
            output_dir=str(
                Path(self.results_dir) / "federated"
            ),
            filename=f"{self.bank_id}_full_trajectory"
        )