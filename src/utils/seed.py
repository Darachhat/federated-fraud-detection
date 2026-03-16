"""
Reproducibility seed control.
Sets seeds for all relevant random number generators.
"""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for full experiment reproducibility.

    Args:
        seed: Integer seed value. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import xgboost as xgb  # noqa: F401
        # XGBoost seed is set via hyperparameter: random_state=seed
    except ImportError:
        pass

    print(f"[SEED] Global seed set to {seed} for full reproducibility.")