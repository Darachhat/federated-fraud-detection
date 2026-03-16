"""
JSON Tree Concatenation Algorithm.
Core technical contribution of the federated framework.

Aggregates locally trained XGBoost models by directly
concatenating their JSON-serialized internal tree structures
at the Global Server, without accessing any raw client data.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List

import xgboost as xgb


class JSONTreeConcatenator:
    """
    Implements the JSON Tree Concatenation Algorithm for
    horizontal Federated Learning with XGBoost.

    Theoretical basis:
        XGBoost prediction = Σ f_k(x) for all trees k.
        Concatenating tree ensembles from multiple clients
        preserves additive scoring semantics while incorporating
        fraud-discriminative patterns from all client distributions.
    """

    def __init__(self):
        self._tree_count_log: List[Dict] = []

    def extract_trees(self, model_path: str) -> List[Dict]:
        """
        Extract the tree array from a serialized XGBoost JSON model.

        Args:
            model_path: Path to the client's JSON model file.

        Returns:
            List of tree dictionaries from the model's internal
            gradient booster representation.
        """
        assert Path(model_path).exists(), \
            f"Model file not found: {model_path}"

        with open(model_path, "r") as f:
            model_json = json.load(f)

        trees = (
            model_json["learner"]
            ["gradient_booster"]
            ["model"]
            ["trees"]
        )

        print(
            f"[AGGREGATOR] Extracted {len(trees)} trees "
            f"from {Path(model_path).name}"
        )
        return trees

    def concatenate(
        self,
        model_paths: List[str]
    ) -> List[Dict]:
        """
        Concatenate tree arrays from multiple client models
        into a single unified tree list.

        Args:
            model_paths: Ordered list of client JSON model paths.

        Returns:
            Concatenated list of all trees with updated indices.
        """
        all_trees = []
        tree_index = 0

        for path in model_paths:
            client_trees = self.extract_trees(path)
            for tree in client_trees:
                tree_copy = copy.deepcopy(tree)
                tree_copy["id"] = tree_index
                all_trees.append(tree_copy)
                tree_index += 1

        print(
            f"[AGGREGATOR] Total concatenated trees: "
            f"{len(all_trees)} "
            f"(from {len(model_paths)} clients)"
        )
        return all_trees

    def build_federated_model(
        self,
        base_model_path: str,
        concatenated_trees: List[Dict],
        output_path: str
    ) -> None:
        """
        Inject the concatenated tree list into a base model
        structure and save the federated global model.

        Args:
            base_model_path:    Path to any client model used
                                as the structural template.
            concatenated_trees: Output of concatenate().
            output_path:        Destination path for the
                                federated global model JSON.
        """
        with open(base_model_path, "r") as f:
            fed_model = json.load(f)

        # Inject concatenated trees
        (
            fed_model["learner"]
            ["gradient_booster"]
            ["model"]
            ["trees"]
        ) = concatenated_trees

        # Update tree count metadata
        (
            fed_model["learner"]
            ["gradient_booster"]
            ["model"]
            ["gbtree_model_param"]
            ["num_trees"]
        ) = str(len(concatenated_trees))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(fed_model, f, indent=2)

        print(
            f"[AGGREGATOR] Federated model saved → {output_path} "
            f"({len(concatenated_trees)} trees total)"
        )

    def aggregate(
        self,
        client_model_paths: List[str],
        base_model_path: str,
        output_path: str
    ) -> str:
        """
        Full JSON Tree Concatenation Algorithm pipeline.

        Steps:
            1. Extract tree arrays from all client models.
            2. Concatenate into a unified tree list.
            3. Inject into base model structure.
            4. Save federated global model to output_path.

        Args:
            client_model_paths: Paths to all client model JSON files.
            base_model_path:    Template model for JSON structure.
            output_path:        Output path for federated model.

        Returns:
            Path to the saved federated global model.
        """
        print(
            f"\n{'═' * 50}\n"
            f"  JSON Tree Concatenation Algorithm\n"
            f"  Aggregating {len(client_model_paths)} client models\n"
            f"{'═' * 50}"
        )

        concatenated_trees = self.concatenate(client_model_paths)
        self.build_federated_model(
            base_model_path=base_model_path,
            concatenated_trees=concatenated_trees,
            output_path=output_path
        )

        return output_path