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
SAME_METRIC_FILE = RESULTS_DIR / "bank_leaf_expansion_same_metric_summary.json"
OUTPUT_FILE = RESULTS_DIR / "bank_leaf_expansion_optimization_priority.json"


def _read_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _build_base_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, random_state=42)
    model.fit(X_train, y_train)
    return model


def _evaluate_feature(
    leaf_X_train: np.ndarray,
    leaf_y_train: np.ndarray,
    leaf_X_test: np.ndarray,
    leaf_y_test: np.ndarray,
    feature_id: int,
) -> dict:
    stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=42)
    stump.fit(leaf_X_train[:, [feature_id]], leaf_y_train)
    train_accuracy = float(np.mean(stump.predict(leaf_X_train[:, [feature_id]]) == leaf_y_train))
    test_accuracy = float(np.mean(stump.predict(leaf_X_test[:, [feature_id]]) == leaf_y_test)) if len(leaf_y_test) else 0.0
    return {
        "feature_id": int(feature_id),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def _format_prompt_focus(
    online_feature_name: str,
    best_feature_name: str,
    online_gain: float,
) -> str:
    if online_gain <= 0:
        return "当前在线选择已接近局部最优，优先级较低，可暂不继续写专用 prompt。"
    if online_feature_name == best_feature_name:
        return f"继续围绕 {best_feature_name} 强化稳定性，重点验证是否能把局部收益转成全局收益。"
    return (
        f"优先做 {online_feature_name} 与 {best_feature_name} 的 pairwise / counterfactual prompt，"
        f"目标是把在线排序从 {online_feature_name} 翻到 {best_feature_name}。"
    )


def build_priority_report() -> dict:
    feature_parts, target_parts, info = load_mnist_dataset("bank", "example_datasets")
    X_train = _merge_train_val(feature_parts)
    y_train = _merge_train_val(target_parts).astype(int)
    X_test = feature_parts["test"]
    y_test = target_parts["test"].astype(int)

    mock_summary = _read_json(MOCK_SUMMARY_FILE)
    online_summary = _read_json(ONLINE_SUMMARY_FILE)
    same_metric_summary = _read_json(SAME_METRIC_FILE)

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

    total_test_samples = int(len(y_test))
    priority_targets = []

    for leaf_id in online_summary.get("selected_leaf_ids", []):
        leaf_id = int(leaf_id)
        context = context_by_leaf.get(leaf_id)
        if context is None:
            continue

        train_idx = np.where(train_leaf_ids == leaf_id)[0]
        test_idx = np.where(test_leaf_ids == leaf_id)[0]
        leaf_X_train = X_train[train_idx]
        leaf_y_train = y_train[train_idx]
        leaf_X_test = X_test[test_idx]
        leaf_y_test = y_test[test_idx]
        feature_by_id = {int(item["feature_id"]): item for item in context.candidate_feature_summaries}

        candidate_scores = [
            _evaluate_feature(leaf_X_train, leaf_y_train, leaf_X_test, leaf_y_test, int(feature_id))
            for feature_id in context.candidate_feature_ids
        ]
        candidate_scores.sort(key=lambda item: (item["test_accuracy"], item["train_accuracy"]), reverse=True)

        best_candidate = candidate_scores[0]
        online_feature = int(online_summary["selected_expansion_features_by_leaf"][str(leaf_id)])
        mock_feature = int(mock_summary["selected_expansion_features_by_leaf"][str(leaf_id)])
        online_choice = next(item for item in candidate_scores if int(item["feature_id"]) == online_feature)
        mock_choice = next(item for item in candidate_scores if int(item["feature_id"]) == mock_feature)

        online_feature_name = str(feature_by_id.get(online_feature, {}).get("feature_name", online_feature))
        mock_feature_name = str(feature_by_id.get(mock_feature, {}).get("feature_name", mock_feature))
        best_feature_id = int(best_candidate["feature_id"])
        best_feature_name = str(feature_by_id.get(best_feature_id, {}).get("feature_name", best_feature_id))
        local_gain = round(best_candidate["test_accuracy"] - online_choice["test_accuracy"], 4)
        estimated_global_gain = round(local_gain * (len(test_idx) / total_test_samples), 6)

        priority_targets.append(
            {
                "priority_score": estimated_global_gain,
                "leaf_id": leaf_id,
                "sample_count_train": int(len(train_idx)),
                "sample_count_test": int(len(test_idx)),
                "predicted_class": int(context.predicted_class),
                "online_selected_feature": online_feature,
                "online_selected_feature_name": online_feature_name,
                "online_local_test_accuracy": round(online_choice["test_accuracy"], 4),
                "mock_selected_feature": mock_feature,
                "mock_selected_feature_name": mock_feature_name,
                "mock_local_test_accuracy": round(mock_choice["test_accuracy"], 4),
                "best_candidate_feature": best_feature_id,
                "best_candidate_feature_name": best_feature_name,
                "best_candidate_local_test_accuracy": round(best_candidate["test_accuracy"], 4),
                "online_to_best_local_gap": local_gain,
                "estimated_global_accuracy_gain": estimated_global_gain,
                "top3_candidate_features": [int(item["feature_id"]) for item in candidate_scores[:3]],
                "top3_candidate_feature_names": [
                    str(feature_by_id.get(int(item["feature_id"]), {}).get("feature_name", item["feature_id"]))
                    for item in candidate_scores[:3]
                ],
                "recommended_prompt_focus": _format_prompt_focus(
                    online_feature_name=online_feature_name,
                    best_feature_name=best_feature_name,
                    online_gain=local_gain,
                ),
            }
        )

    priority_targets.sort(
        key=lambda item: (item["estimated_global_accuracy_gain"], item["sample_count_test"]),
        reverse=True,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "bank",
        "goal": "Prioritize which leaf to optimize next so online_llm can try to move from matching mock_llm to exceeding it.",
        "baseline_test_accuracy": same_metric_summary["experiments"]["baseline_rf"]["test_accuracy"],
        "online_test_accuracy": same_metric_summary["experiments"]["online_llm"]["test_accuracy"],
        "mock_test_accuracy": same_metric_summary["experiments"]["llm_guided"]["test_accuracy"],
        "selected_leaf_ids": [int(leaf_id) for leaf_id in online_summary.get("selected_leaf_ids", [])],
        "priority_targets": priority_targets,
        "summary": {
            "total_selected_leaf_count": len(priority_targets),
            "highest_priority_leaf": priority_targets[0]["leaf_id"] if priority_targets else None,
            "highest_priority_estimated_gain": priority_targets[0]["estimated_global_accuracy_gain"] if priority_targets else 0.0,
            "leaves_with_positive_headroom": sum(1 for item in priority_targets if item["estimated_global_accuracy_gain"] > 0),
        },
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = build_priority_report()
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bank optimization priority report to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
