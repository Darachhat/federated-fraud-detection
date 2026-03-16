"""
File I/O utilities for model serialization,
JSON manipulation, and results persistence.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Dict, path: str) -> None:
    """Serialize a dictionary to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    """Load a JSON file into a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: str
) -> None:
    """
    Persist experiment results to a JSON file.

    Args:
        results:    Dictionary of metric names to values.
        output_dir: Directory to write results file.
        filename:   Output filename (without extension).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{filename}.json"
    save_json(results, str(output_path))
    print(f"[IO] Results saved to {output_path}")


def wait_for_file(
    path: str,
    timeout_seconds: int = 300,
    poll_interval: float = 2.0
) -> bool:
    """
    Block until a file exists or timeout is reached.
    Used for federated round synchronization.

    Args:
        path:             File path to wait for.
        timeout_seconds:  Maximum wait time in seconds.
        poll_interval:    Seconds between existence checks.

    Returns:
        True if file appeared within timeout, False otherwise.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if Path(path).exists():
            return True
        time.sleep(poll_interval)
    return False


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)