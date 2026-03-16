"""
Global Server — Federated Learning Coordinator.
Orchestrates all federation rounds and applies
the JSON Tree Concatenation Algorithm for aggregation.
"""

from pathlib import Path
from typing import Dict, List

from src.federated.aggregator import JSONTreeConcatenator
from src.utils.io_utils import load_config, save_results, wait_for_file
from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed


class FederatedServer:
    """
    Global Server for the Federated Learning framework.

    Responsibilities:
        - Coordinate communication rounds
        - Wait for all client model submissions
        - Aggregate client models via JSON Tree Concatenation
        - Distribute federated global model to all clients
        - Log per-round aggregation metrics
    """

    def __init__(self, config_path: str = "config/base_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "server",
            log_dir=self.config["logging"]["log_dir"]
        )
        self.aggregator = JSONTreeConcatenator()
        self.num_rounds = self.config["federated"]["num_rounds"]
        self.num_clients = self.config["federated"]["num_clients"]
        self.exchange_dir = self.config["federated"]["exchange_dir"]
        self.global_dir = self.config["federated"]["global_model_dir"]
        self.timeout = self.config["federated"]["round_timeout_seconds"]
        self.results_dir = self.config["results"]["output_dir"]

        set_global_seed(self.config["project"]["seed"])

        Path(self.exchange_dir).mkdir(parents=True, exist_ok=True)
        Path(self.global_dir).mkdir(parents=True, exist_ok=True)

    def _client_model_path(self, bank_id: str, round_num: int) -> str:
        """Construct the expected path for a client's round submission."""
        return str(
            Path(self.exchange_dir)
            / f"{bank_id}_round_{round_num}.json"
        )

    def _global_model_path(self, round_num: int) -> str:
        """Construct the output path for the global model."""
        return str(
            Path(self.global_dir)
            / f"global_model_round_{round_num}.json"
        )

    def _wait_for_all_clients(
        self,
        bank_ids: List[str],
        round_num: int
    ) -> List[str]:
        """
        Block until all clients have submitted their round models.

        Returns:
            List of paths to collected client model files.
        """
        collected = []
        for bank_id in bank_ids:
            path = self._client_model_path(bank_id, round_num)
            self.logger.info(
                f"Waiting for {bank_id} | Round {round_num}..."
            )
            arrived = wait_for_file(path, self.timeout)
            if not arrived:
                raise TimeoutError(
                    f"[SERVER] Timeout waiting for {bank_id} "
                    f"at round {round_num}."
                )
            self.logger.info(f"Received: {bank_id} ✓")
            collected.append(path)
        return collected

    def run(self, bank_ids: List[str] = None) -> None:
        """
        Execute the full federated training protocol
        across all configured rounds.

        Args:
            bank_ids: List of client bank identifiers.
                      Defaults to ['bank1', 'bank2', 'bank3'].
        """
        if bank_ids is None:
            bank_ids = ["bank1", "bank2", "bank3"]

        self.logger.info(
            f"Federated Server started. "
            f"Clients: {bank_ids} | Rounds: {self.num_rounds}"
        )

        round_results = []

        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(
                f"\n{'═' * 50}\n"
                f"  FEDERATION ROUND {round_num} / {self.num_rounds}\n"
                f"{'═' * 50}"
            )

            # Step 1 — Collect client models
            client_paths = self._wait_for_all_clients(
                bank_ids, round_num
            )

            # Step 2 — Aggregate via JSON Tree Concatenation
            global_model_path = self._global_model_path(round_num)
            self.aggregator.aggregate(
                client_model_paths = client_paths,
                base_model_path    = client_paths[0],
                output_path        = global_model_path
            )

            self.logger.info(
                f"Round {round_num} aggregation complete. "
                f"Global model → {global_model_path}"
            )

            round_results.append({
                "round": round_num,
                "global_model_path": global_model_path,
                "clients_aggregated": len(client_paths)
            })

            # Step 3 — Clean up exchange directory for next round
            for path in client_paths:
                Path(path).unlink(missing_ok=True)

        # Save server-side round log
        save_results(
            {"rounds": round_results},
            output_dir=str(
                Path(self.results_dir)
                / "federated"
            ),
            filename="server_round_log"
        )

        self.logger.info("All federation rounds complete.")