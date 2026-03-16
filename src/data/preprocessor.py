"""
Feature engineering and preprocessing pipeline
for the PaySim synthetic financial dataset.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PaySimPreprocessor:
    """
    Preprocessing pipeline for the PaySim dataset.

    Responsibilities:
        - Drop non-informative identifier columns
        - Remove legacy rule-based flag feature
        - One-hot encode transaction type
        - Engineer balance discrepancy features
        - Produce stratified train/test splits
    """

    # Columns to drop before modeling
    DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]

    # Target column
    TARGET = "isFraud"

    def __init__(self, test_size: float = 0.20, seed: int = 42):
        """
        Args:
            test_size: Proportion of data reserved for testing.
            seed:      Random seed for reproducibility.
        """
        self.test_size = test_size
        self.seed = seed

    def load(self, path: str) -> pd.DataFrame:
        """Load raw PaySim CSV into a DataFrame."""
        df = pd.read_csv(path)
        print(f"[PREPROCESSOR] Loaded {len(df):,} records from {path}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-informative columns and validate target."""
        df = df.drop(
            columns=[c for c in self.DROP_COLUMNS if c in df.columns]
        )
        assert self.TARGET in df.columns, \
            f"Target column '{self.TARGET}' not found."
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct domain-informed engineered features.

        Features:
            errorBalanceOrig: Discrepancy in originating account balance.
            errorBalanceDest: Discrepancy in destination account balance.
        """
        df = df.copy()

        df["errorBalanceOrig"] = (
            df["newbalanceOrig"]
            - (df["oldbalanceOrg"] - df["amount"])
        )

        df["errorBalanceDest"] = (
            df["newbalanceDest"]
            - (df["oldbalanceDest"] + df["amount"])
        )

        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the transaction type feature."""
        df = pd.get_dummies(df, columns=["type"], prefix="type")
        return df

    def split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produce a stratified train/test split.

        Returns:
            Tuple of (train_df, test_df).
        """
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[self.TARGET],
            random_state=self.seed
        )
        print(
            f"[PREPROCESSOR] Split → "
            f"Train: {len(train_df):,} | Test: {len(test_df):,}"
        )
        return train_df, test_df

    def run(
        self,
        path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full preprocessing pipeline.

        Returns:
            Tuple of (train_df, test_df).
        """
        df = self.load(path)
        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df)
        train_df, test_df = self.split(df)
        return train_df, test_df