from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.leaf_expansion import SUPPORTED_LEAF_SELECTION_STRATEGIES, select_candidate_leaves  # noqa: E402
from run_leaf_expansion_mnist import _merge_train_val, load_mnist_dataset  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "bank"
OUTPUT_FILE = RESULTS_DIR / "bank_leaf_selection_strategy_preview.json"


def _build_base_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_strategy_preview() -> dict:
    feature_parts, target_parts, info = load_mnist_dataset("bank", "example_datasets")
    X_train = _merge_train_val(feature_parts)
    y_train = _merge_train_val(target_parts).astype(int)

    base_tree = _build_base_tree(X_train, y_train)
    strategies: dict[str, list[dict[str, object]]] = {}

    for strategy in SUPPORTED_LEAF_SELECTION_STRATEGIES:
        contexts = select_candidate_leaves(
            base_tree,
            X_train,
            y_train,
            top_k_leaves=3,
            min_leaf_samples=80,
            top_feature_k=10,
            leaf_selection_strategy=strategy,
            feature_names=info.get("leaf_expansion_feature_names"),
            feature_descriptions=info.get("leaf_expansion_feature_descriptions"),
            feature_types=info.get("leaf_expansion_feature_types"),
            feature_value_hints=info.get("leaf_expansion_feature_value_hints"),
            prompt_metadata=info.get("leaf_expansion_prompt_metadata"),
        )
        strategies[strategy] = [
            {
                "leaf_id": int(context.leaf_id),
                "sample_count": int(context.sample_count),
                "impurity": float(context.impurity),
                "predicted_class": int(context.predicted_class),
                "top_candidate_features": [int(feature_id) for feature_id in context.candidate_feature_ids[:3]],
            }
            for context in contexts
        ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "bank",
        "goal": "Preview how different candidate leaf selection strategies would change the next bank experiments.",
        "strategies": strategies,
        "summary": {
            "default_strategy": "impurity_mass",
            "available_strategies": list(SUPPORTED_LEAF_SELECTION_STRATEGIES),
            "default_leaf_ids": [item["leaf_id"] for item in strategies["impurity_mass"]],
            "sample_count_leaf_ids": [item["leaf_id"] for item in strategies["sample_count"]],
            "impurity_leaf_ids": [item["leaf_id"] for item in strategies["impurity"]],
        },
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = build_strategy_preview()
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bank leaf selection strategy preview to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
