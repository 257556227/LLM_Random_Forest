from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_leaf_expansion_mnist import _merge_train_val, load_mnist_dataset  # noqa: E402
from model.leaf_expansion import select_candidate_leaves  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
OUTPUT_FILE = RESULTS_DIR / "mnist_leaf_expansion_oracle_analysis.json"


def _load_summary(file_name: str) -> dict:
    return json.loads((RESULTS_DIR / file_name).read_text(encoding="utf-8"))


def _single_feature_scores(leaf_X_train, leaf_y_train, leaf_X_test, leaf_y_test, candidate_ids: list[int]) -> list[dict]:
    scores = []
    for feature_id in candidate_ids:
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=42)
        stump.fit(leaf_X_train[:, [feature_id]], leaf_y_train)
        train_accuracy = accuracy_score(leaf_y_train, stump.predict(leaf_X_train[:, [feature_id]]))
        test_accuracy = accuracy_score(leaf_y_test, stump.predict(leaf_X_test[:, [feature_id]]))
        scores.append(
            {
                "feature_id": int(feature_id),
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
            }
        )
    return sorted(scores, key=lambda item: (item["test_accuracy"], item["train_accuracy"]), reverse=True)


def build_oracle_analysis() -> dict:
    numeric_parts, target_parts, _ = load_mnist_dataset("mnist", "example_datasets")
    X_train = _merge_train_val(numeric_parts)
    y_train = _merge_train_val(target_parts).astype(int)
    X_test = numeric_parts["test"]
    y_test = target_parts["test"].astype(int)

    base_tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, random_state=42)
    base_tree.fit(X_train, y_train)

    selected_contexts = select_candidate_leaves(
        base_tree,
        X_train,
        y_train,
        top_k_leaves=3,
        min_leaf_samples=80,
        top_feature_k=10,
    )
    train_leaf_ids = base_tree.apply(X_train)
    test_leaf_ids = base_tree.apply(X_test)

    without_summary = _load_summary("leaf_expansion_without_llm_summary.json")
    mock_summary = _load_summary("llm_guided_leaf_expansion_summary.json")
    online_summary = _load_summary("llm_guided_leaf_expansion_online_summary.json")

    leaves = {}
    for context in selected_contexts:
        leaf_key = str(context.leaf_id)
        train_idx = np.where(train_leaf_ids == context.leaf_id)[0]
        test_idx = np.where(test_leaf_ids == context.leaf_id)[0]
        leaf_X_train = X_train[train_idx]
        leaf_y_train = y_train[train_idx]
        leaf_X_test = X_test[test_idx]
        leaf_y_test = y_test[test_idx]

        oracle_scores = _single_feature_scores(
            leaf_X_train,
            leaf_y_train,
            leaf_X_test,
            leaf_y_test,
            context.candidate_feature_ids,
        )
        oracle_best = oracle_scores[0]

        method_rows = {}
        for method_name, summary in (
            ("without_llm", without_summary),
            ("mock_llm", mock_summary),
            ("online_llm", online_summary),
        ):
            chosen_feature_id = int(summary["selected_expansion_features_by_leaf"][leaf_key])
            chosen_score = next(
                item for item in oracle_scores if int(item["feature_id"]) == chosen_feature_id
            )
            method_rows[method_name] = {
                "chosen_feature_id": chosen_feature_id,
                "local_test_accuracy": chosen_score["test_accuracy"],
                "oracle_gap": round(oracle_best["test_accuracy"] - chosen_score["test_accuracy"], 4),
            }

        leaves[leaf_key] = {
            "sample_count_train": int(len(train_idx)),
            "sample_count_test": int(len(test_idx)),
            "oracle_best_feature": oracle_best,
            "top3_oracle_features": oracle_scores[:3],
            "methods": method_rows,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "mnist",
        "selected_leaf_ids": [int(context.leaf_id) for context in selected_contexts],
        "focus_leaf_ids": [8, 9],
        "leaves": leaves,
        "summary": {
            "mock_accuracy": float(mock_summary["expanded_accuracy"]),
            "online_accuracy": float(online_summary["expanded_accuracy"]),
            "online_minus_mock": round(float(online_summary["expanded_accuracy"]) - float(mock_summary["expanded_accuracy"]), 4),
        },
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = build_oracle_analysis()
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote oracle analysis to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()