"""
Evaluation metrics for fraud detection models.
AUPRC and F1-Score only. Accuracy is explicitly excluded.
"""

from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score
)


TARGET = "isFraud"


def evaluate_model(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
    bank_id: str = "unknown",
    round_num: int = 0
) -> Dict[str, float]:
    """
    Evaluate a trained XGBoost model using AUPRC and F1-Score.
    Accuracy is intentionally excluded from all evaluations.

    Args:
        model:      Trained XGBClassifier instance.
        test_df:    Test DataFrame with TARGET column.
        threshold:  Classification threshold for F1 calculation.
        bank_id:    Identifier string for logging.
        round_num:  Current federation round number.

    Returns:
        Dictionary containing:
            - auprc
            - f1_score
            - precision
            - recall
            - threshold
    """
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Predicted probabilities for AUPRC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Predicted labels for F1
    y_pred = (y_prob >= threshold).astype(int)

    # AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test, y_prob
    )
    auprc = auc(recall_vals, precision_vals)

    # F1-Score
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Supporting metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    results = {
        "bank_id": bank_id,
        "round": round_num,
        "auprc": round(float(auprc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "threshold": threshold,
        # NOTE: Accuracy intentionally omitted.
        # It is a misleading metric under 0.13% class imbalance.
    }

    _print_results(results)
    return results


def _print_results(results: Dict) -> None:
    """Pretty-print evaluation results to console."""
    print(
        f"\n{'─' * 50}\n"
        f"  [{results['bank_id'].upper()}] "
        f"Round {results['round']} Evaluation\n"
        f"{'─' * 50}\n"
        f"  AUPRC     : {results['auprc']:.4f}\n"
        f"  F1-Score  : {results['f1_score']:.4f}\n"
        f"  Precision : {results['precision']:.4f}\n"
        f"  Recall    : {results['recall']:.4f}\n"
        f"  Threshold : {results['threshold']}\n"
        f"  [Accuracy : EXCLUDED — misleading under 0.13% imbalance]\n"
        f"{'─' * 50}\n"
    )