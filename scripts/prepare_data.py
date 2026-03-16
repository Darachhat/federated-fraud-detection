"""
One-time data preparation script.
Run once before any experiments.

Usage:
    python scripts/prepare_data.py
"""

from pathlib import Path

from src.data.partitioner import NonIIDPartitioner
from src.data.preprocessor import PaySimPreprocessor
from src.utils.io_utils import load_config
from src.utils.seed import set_global_seed

CONFIG_PATH = "config/base_config.yaml"


def main():
    config = load_config(CONFIG_PATH)
    set_global_seed(config["project"]["seed"])

    print("\n" + "═" * 60)
    print("  DATA PREPARATION PIPELINE")
    print("═" * 60 + "\n")

    # Step 1 — Preprocess raw PaySim dataset
    preprocessor = PaySimPreprocessor(
        test_size=config["data"]["test_size"],
        seed=config["project"]["seed"]
    )
    train_df, test_df = preprocessor.run(config["data"]["raw_path"])

    # Step 2 — Save global test set (shared across all experiments)
    processed_dir = Path(config["data"]["processed_path"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    global_test_path = processed_dir / "global_test.csv"
    test_df.to_csv(global_test_path, index=False)
    print(f"[PREPARE] Global test set saved → {global_test_path}")

    # Step 3 — Non-IID partitioning
    partitioner = NonIIDPartitioner(
        seed=config["project"]["seed"]
    )
    bank1_df, bank2_df, bank3_df = partitioner.partition(train_df)

    # Step 4 — Save per-bank training partitions
    for bank_id, df in [
        ("bank1", bank1_df),
        ("bank2", bank2_df),
        ("bank3", bank3_df)
    ]:
        train_path = processed_dir / f"{bank_id}_train.csv"
        test_path = processed_dir / f"{bank_id}_test.csv"

        # Local test = 20% of local training data
        local_test_size = int(len(df) * 0.2)
        local_test = df.sample(
            n=local_test_size,
            random_state=config["project"]["seed"]
        )
        local_train = df.drop(index=local_test.index)

        local_train.to_csv(train_path, index=False)
        local_test.to_csv(test_path, index=False)

        print(
            f"[PREPARE] {bank_id}: "
            f"Train → {train_path} | "
            f"Test → {test_path}"
        )

    print("\n[PREPARE] Data preparation complete. ✓\n")


if __name__ == "__main__":
    main()