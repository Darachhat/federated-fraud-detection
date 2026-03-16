"""
Launch a federated client bank (Terminals 1, 2, or 3).

Usage:
    Terminal 1:  python scripts/run_client.py --bank bank1
    Terminal 2:  python scripts/run_client.py --bank bank2
    Terminal 3:  python scripts/run_client.py --bank bank3
"""

import argparse

from src.federated.client import FederatedClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a federated client bank."
    )
    parser.add_argument(
        "--bank",
        type=str,
        required=True,
        choices=["bank1", "bank2", "bank3"],
        help="Bank identifier to launch as federated client."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_map = {
        "bank1": "config/bank1_config.yaml",
        "bank2": "config/bank2_config.yaml",
        "bank3": "config/bank3_config.yaml",
    }
    client = FederatedClient(
        bank_config_path=config_map[args.bank],
        base_config_path="config/base_config.yaml"
    )
    client.run()


if __name__ == "__main__":
    main()