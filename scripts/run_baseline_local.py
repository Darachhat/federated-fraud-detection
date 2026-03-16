"""
Local-Only Baseline Experiment.
Trains isolated XGBoost models for each bank
with zero federation or data sharing.

Usage:
    python scripts/run_baseline_local.py
"""

from pathlib import Path

from src.evaluation.metrics import evaluate_model
from src.models.xgboost_trainer import XGBoostTrainer
from src.utils.io_utils import load_config, load_dataframe, save_results
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

    print("\n" + "═" * 60)
    print("  LOCAL-ONLY BASELINE EXPERIMENT")
    print("═" * 60 + "\n")

    all_results = []

    for bank_cfg_path in BANK_CONFIGS:
        bank_config = load_config(bank_cfg_path)
        bank_id = bank_config["client"]["bank_id"]

        print(f"\n[BASELINE] Training {bank_id.upper()} local model...")

        # Load data
        train_df = load_dataframe(bank_config["data"]["train_path"])
        test_df = load_dataframe(
            bank_config["data"]["global_test_path"]
        )

        # Train local model
        trainer = XGBoostTrainer(base_config)
        trainer.train(train_df)

        # Save local model
        model_path = str(
            Path(base_config["federated"]["local_model_dir"])
            / f"{bank_id}_local_model.json"
        )
        trainer.save(model_path)

        # Evaluate on global test set
        results = evaluate_model(
            model=trainer.model,
            test_df=test_df,
            threshold=base_config["evaluation"]["threshold"],
            bank_id=bank_id,
            round_num=0
        )
        all_results.append(results)

        # Save individual bank results
        save_results(
            results,
            output_dir=str(
                Path(base_config["results"]["output_dir"])
                / "local_baseline"
            ),
            filename=f"{bank_id}_local_baseline"
        )

    # Save consolidated results
    save_results(
        {"local_baseline": all_results},
        output_dir=str(
            Path(base_config["results"]["output_dir"])
            / "local_baseline"
        ),
        filename="all_banks_local_baseline"
    )

    print("\n[BASELINE] Local-Only experiment complete. ✓\n")


if __name__ == "__main__":
    main()