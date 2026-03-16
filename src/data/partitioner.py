"""
Non-IID dataset partitioner.
Splits the global PaySim training set into three
client-specific subsets with distinct fraud distributions.
"""

from typing import Dict, Tuple

import pandas as pd


class NonIIDPartitioner:
    """
    Partitions the PaySim training dataset into three
    Non-IID client subsets simulating distinct bank profiles:

        Bank 1 — High-Risk:   TRANSFER + CASH_OUT, fraud-enriched
        Bank 2 — Retail:      PAYMENT + CASH_IN,   zero fraud labels
        Bank 3 — Mixed:       Remaining transactions, moderate fraud
    """

    TARGET = "isFraud"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def partition(
        self,
        train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition the training DataFrame into three Non-IID subsets.

        Args:
            train_df: Full preprocessed training DataFrame.

        Returns:
            Tuple of (bank1_df, bank2_df, bank3_df).
        """
        # Reconstruct type column from one-hot encoding
        type_cols = [c for c in train_df.columns if c.startswith("type_")]

        # Helper: get transaction type label from one-hot row
        def get_type(row):
            for col in type_cols:
                if row[col] == 1:
                    return col.replace("type_", "")
            return "UNKNOWN"

        df = train_df.copy()
        df["_type"] = df.apply(get_type, axis=1)

        # ── Bank 2: Retail — PAYMENT and CASH_IN only (zero fraud) ──
        bank2_mask = df["_type"].isin(["PAYMENT", "CASH_IN"])
        bank2_df = df[bank2_mask].copy()

        # Force zero fraud labels for Bank 2
        bank2_df[self.TARGET] = 0

        # ── Bank 1: High-Risk — TRANSFER and CASH_OUT ──
        high_risk_mask = df["_type"].isin(["TRANSFER", "CASH_OUT"])
        high_risk_df = df[high_risk_mask & ~bank2_mask].copy()

        # Sample 60% for Bank 1, remainder goes to Bank 3
        bank1_df = high_risk_df.sample(
            frac=0.60,
            random_state=self.seed
        )
        bank3_high_risk = high_risk_df.drop(index=bank1_df.index)

        # ── Bank 3: Mixed — remaining transactions ──
        remaining_mask = ~bank2_mask & ~high_risk_mask
        remaining_df = df[remaining_mask].copy()
        bank3_df = pd.concat(
            [bank3_high_risk, remaining_df],
            ignore_index=True
        )

        # Drop helper column
        for partition in [bank1_df, bank2_df, bank3_df]:
            partition.drop(columns=["_type"], inplace=True)

        self._log_partition_stats(bank1_df, bank2_df, bank3_df)

        return bank1_df, bank2_df, bank3_df

    def _log_partition_stats(
        self,
        bank1_df: pd.DataFrame,
        bank2_df: pd.DataFrame,
        bank3_df: pd.DataFrame
    ) -> None:
        """Print partition statistics to console."""
        for name, df in [
            ("Bank 1 (High-Risk)", bank1_df),
            ("Bank 2 (Retail/Blind)", bank2_df),
            ("Bank 3 (Mixed)", bank3_df),
        ]:
            total = len(df)
            fraud = df[self.TARGET].sum()
            pct = (fraud / total * 100) if total > 0 else 0.0
            print(
                f"[PARTITION] {name}: "
                f"{total:>10,} transactions | "
                f"{int(fraud):>6,} fraud | "
                f"{pct:.4f}% fraud prevalence"
            )