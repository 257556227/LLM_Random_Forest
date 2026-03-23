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

from model.leaf_expansion import select_candidate_leaves  # noqa: E402
from run_leaf_expansion_mnist import _merge_train_val, load_mnist_dataset  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "bank"
MOCK_SUMMARY_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_summary.json"
ONLINE_SUMMARY_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
OUTPUT_FILE = RESULTS_DIR / "bank_leaf_expansion_online_vs_mock_leaf_compare.json"


def _build_base_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    base_tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, random_state=42)
    base_tree.fit(X_train, y_train)
    return base_tree


def _evaluate_single_feature(
    leaf_X_train: np.ndarray,
    leaf_y_train: np.ndarray,
    leaf_X_test: np.ndarray,
    leaf_y_test: np.ndarray,
    feature_id: int,
) -> tuple[float, float]:
    stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=42)
    stump.fit(leaf_X_train[:, [feature_id]], leaf_y_train)
    train_accuracy = float(np.mean(stump.predict(leaf_X_train[:, [feature_id]]) == leaf_y_train))
    test_accuracy = float(np.mean(stump.predict(leaf_X_test[:, [feature_id]]) == leaf_y_test)) if len(leaf_y_test) else 0.0
    return train_accuracy, test_accuracy


def _read_summary(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def build_comparison() -> dict:
    feature_parts, target_parts, info = load_mnist_dataset("bank", "example_datasets")
    X_train = _merge_train_val(feature_parts)
    y_train = _merge_train_val(target_parts).astype(int)
    X_test = feature_parts["test"]
    y_test = target_parts["test"].astype(int)

    mock_summary = _read_summary(MOCK_SUMMARY_FILE)
    online_summary = _read_summary(ONLINE_SUMMARY_FILE)

    base_tree = _build_base_tree(X_train, y_train)
    train_leaf_ids = base_tree.apply(X_train)
    test_leaf_ids = base_tree.apply(X_test)
    contexts = select_candidate_leaves(
        base_tree,
        X_train,
        y_train,
        top_k_leaves=3,
        min_leaf_samples=80,
        top_feature_k=10,
        feature_names=info.get("leaf_expansion_feature_names"),
        feature_descriptions=info.get("leaf_expansion_feature_descriptions"),
        feature_types=info.get("leaf_expansion_feature_types"),
        feature_value_hints=info.get("leaf_expansion_feature_value_hints"),
        prompt_metadata=info.get("leaf_expansion_prompt_metadata"),
    )
    context_by_leaf = {int(context.leaf_id): context for context in contexts}

    leaf_comparisons = []
    online_better_count = 0
    mock_better_count = 0

    for leaf_id in mock_summary.get("selected_leaf_ids", []):
        leaf_id = int(leaf_id)
        if leaf_id not in context_by_leaf:
            continue
        context = context_by_leaf[leaf_id]
        train_indices = np.where(train_leaf_ids == leaf_id)[0]
        test_indices = np.where(test_leaf_ids == leaf_id)[0]
        leaf_X_train = X_train[train_indices]
        leaf_y_train = y_train[train_indices]
        leaf_X_test = X_test[test_indices]
        leaf_y_test = y_test[test_indices]
        feature_by_id = {int(item["feature_id"]): item for item in context.candidate_feature_summaries}

        mock_feature = int(mock_summary["selected_expansion_features_by_leaf"][str(leaf_id)])
        online_feature = int(online_summary["selected_expansion_features_by_leaf"][str(leaf_id)])
        mock_train_acc, mock_test_acc = _evaluate_single_feature(leaf_X_train, leaf_y_train, leaf_X_test, leaf_y_test, mock_feature)
        online_train_acc, online_test_acc = _evaluate_single_feature(leaf_X_train, leaf_y_train, leaf_X_test, leaf_y_test, online_feature)

        if online_test_acc > mock_test_acc:
            online_better_count += 1
        elif mock_test_acc > online_test_acc:
            mock_better_count += 1

        baseline_test_accuracy = float(np.mean(base_tree.predict(leaf_X_test) == leaf_y_test)) if len(leaf_y_test) else 0.0
        mock_ranked = [int(item) for item in mock_summary["ranked_features_by_leaf"].get(str(leaf_id), [])]
        online_ranked = [int(item) for item in online_summary["ranked_features_by_leaf"].get(str(leaf_id), [])]
        overlap_top3 = len(set(mock_ranked[:3]).intersection(online_ranked[:3]))

        leaf_comparisons.append(
            {
                "leaf_id": leaf_id,
                "sample_count_train": int(len(train_indices)),
                "sample_count_test": int(len(test_indices)),
                "predicted_class": int(context.predicted_class),
                "baseline_leaf_test_accuracy": baseline_test_accuracy,
                "mock_selected_feature": mock_feature,
                "mock_selected_feature_name": str(feature_by_id.get(mock_feature, {}).get("feature_name", mock_feature)),
                "mock_selected_feature_description": str(feature_by_id.get(mock_feature, {}).get("feature_description", "")),
                "mock_local_train_accuracy": mock_train_acc,
                "mock_local_test_accuracy": mock_test_acc,
                "online_selected_feature": online_feature,
                "online_selected_feature_name": str(feature_by_id.get(online_feature, {}).get("feature_name", online_feature)),
                "online_selected_feature_description": str(feature_by_id.get(online_feature, {}).get("feature_description", "")),
                "online_local_train_accuracy": online_train_acc,
                "online_local_test_accuracy": online_test_acc,
                "online_minus_mock_local_test_accuracy": round(online_test_acc - mock_test_acc, 4),
                "same_selected_feature": mock_feature == online_feature,
                "mock_ranked_top5": mock_ranked[:5],
                "online_ranked_top5": online_ranked[:5],
                "top3_overlap_count": int(overlap_top3),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "bank",
        "mock_summary_file": str(MOCK_SUMMARY_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "online_summary_file": str(ONLINE_SUMMARY_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "mock_global_test_accuracy": mock_summary.get("expanded_accuracy"),
        "online_global_test_accuracy": online_summary.get("expanded_accuracy"),
        "leaf_comparisons": leaf_comparisons,
        "aggregate": {
            "leaf_count": len(leaf_comparisons),
            "online_better_leaf_count": online_better_count,
            "mock_better_leaf_count": mock_better_count,
            "same_feature_leaf_count": sum(1 for item in leaf_comparisons if item["same_selected_feature"]),
        },
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = build_comparison()
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bank leaf comparison to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()