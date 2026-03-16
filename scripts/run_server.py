"""
Launch the Global Server (Terminal 0).

Usage:
    python scripts/run_server.py
"""

from src.federated.server import FederatedServer


def main():
    server = FederatedServer(
        config_path="config/base_config.yaml"
    )
    server.run(bank_ids=["bank1", "bank2", "bank3"])


if __name__ == "__main__":
    main()