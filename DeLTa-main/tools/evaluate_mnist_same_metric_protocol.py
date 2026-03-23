from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.leaf_expansion import (  # noqa: E402
    fit_leaf_expansion,
    rank_features_with_mock_llm,
    rank_features_without_llm,
    select_candidate_leaves,
)
from run_leaf_expansion_mnist import (  # noqa: E402
    _merge_train_val,
    load_mnist_dataset,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
OUTPUT_FILE = RESULTS_DIR / "mnist_leaf_expansion_same_metric_summary.json"
ONLINE_ATTEMPT_FILE = RESULTS_DIR / "mnist_leaf_expansion_online_llm_attempt.json"


def _load_mainline_result(file_path: Path) -> dict[str, Any]:
    payload = np.load(file_path, allow_pickle=True).item()
    logits = np.asarray(payload["logit"])
    labels = np.asarray(payload["label"]).astype(int)
    predictions = np.argmax(logits, axis=1)
    return {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sample_count": int(labels.shape[0]),
        "test_accuracy": float(accuracy_score(labels, predictions)),
    }


def _build_base_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    base_tree = DecisionTreeClassifier(
        max_depth=3,
        max_leaf_nodes=20,
        random_state=42,
    )
    base_tree.fit(X_train, y_train)
    return base_tree


def _leaf_expansion_same_metric(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    base_tree = _build_base_tree(X_train, y_train)
    leaf_ids = base_tree.apply(X_train)
    selected_contexts = select_candidate_leaves(
        base_tree,
        X_train,
        y_train,
        top_k_leaves=3,
        min_leaf_samples=80,
        top_feature_k=10,
    )

    without_ranked = {}
    llm_ranked = {}
    for context in selected_contexts:
        indices = np.where(leaf_ids == context.leaf_id)[0]
        leaf_X = X_train[indices]
        without_ranked[int(context.leaf_id)] = rank_features_without_llm(
            leaf_X,
            top_feature_ids=context.candidate_feature_ids,
        )
        llm_ranked[int(context.leaf_id)] = rank_features_with_mock_llm(context)

    without_model = fit_leaf_expansion(
        base_tree=base_tree,
        X_train=X_train,
        y_train=y_train,
        selected_leaf_contexts=selected_contexts,
        ranked_features_by_leaf=without_ranked,
        random_state=42,
    )
    guided_model = fit_leaf_expansion(
        base_tree=base_tree,
        X_train=X_train,
        y_train=y_train,
        selected_leaf_contexts=selected_contexts,
        ranked_features_by_leaf=llm_ranked,
        random_state=42,
    )

    without_predictions = without_model.predict(X_test)
    guided_predictions = guided_model.predict(X_test)

    without_summary = {
        "source_mode": "run_leaf_expansion_mnist without_llm equivalent",
        "sample_count": int(y_test.shape[0]),
        "test_accuracy": float(accuracy_score(y_test, without_predictions)),
        "selected_leaf_ids": [int(context.leaf_id) for context in selected_contexts],
    }
    guided_summary = {
        "source_mode": "run_leaf_expansion_mnist llm_guided mock_llm equivalent",
        "sample_count": int(y_test.shape[0]),
        "test_accuracy": float(accuracy_score(y_test, guided_predictions)),
        "selected_leaf_ids": [int(context.leaf_id) for context in selected_contexts],
        "relation_source": "mock_llm",
    }
    return without_summary, guided_summary


def build_summary() -> dict[str, Any]:
    numeric_parts, target_parts, _info = load_mnist_dataset("mnist", "example_datasets")
    X_train = _merge_train_val(numeric_parts)
    y_train = _merge_train_val(target_parts).astype(int)
    X_test = numeric_parts["test"]
    y_test = target_parts["test"].astype(int)

    baseline_rf = _load_mainline_result(RESULTS_DIR / "mnist_md8_ml20_tree3.npy")
    deeper_rf = _load_mainline_result(RESULTS_DIR / "mnist_md20_ml100_tree15.npy")
    without_llm, llm_guided = _leaf_expansion_same_metric(X_train, y_train, X_test, y_test)

    return {
        "protocol_name": "mnist_leaf_expansion_same_metric_v1",
        "dataset": "mnist",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "same_test_accuracy_eval",
        "strict_same_metric_protocol_ready": True,
        "experiments": {
            "baseline_rf": baseline_rf,
            "deeper_rf": deeper_rf,
            "without_llm": without_llm,
            "llm_guided": llm_guided,
        },
        "comparisons": {
            "deeper_minus_baseline": round(deeper_rf["test_accuracy"] - baseline_rf["test_accuracy"], 4),
            "llm_guided_minus_without": round(llm_guided["test_accuracy"] - without_llm["test_accuracy"], 4),
            "llm_guided_minus_baseline": round(llm_guided["test_accuracy"] - baseline_rf["test_accuracy"], 4),
        },
        "notes": [
            "这份摘要只比较统一的 test accuracy，不混用 fused eval 与原型摘要中的单独字段。",
            "baseline_rf 与 deeper_rf 来自主链 .npy 里的 logit/label；without_llm 与 llm_guided 由同一份 MNIST 数据和固定原型参数现场重算。",
            "因此 strict_same_metric_protocol_ready 在这份摘要内为 true，但它不替代库存摘要文件的历史记录价值。",
        ],
    }


def attach_online_llm_attempt(summary: dict[str, Any]) -> dict[str, Any]:
    if not ONLINE_ATTEMPT_FILE.exists():
        return summary

    attempt_payload = json.loads(ONLINE_ATTEMPT_FILE.read_text(encoding="utf-8"))
    summary["online_llm_attempt"] = attempt_payload

    if attempt_payload.get("status") == "success" and attempt_payload.get("test_accuracy") is not None:
        summary["experiments"]["online_llm"] = {
            "source_file": attempt_payload.get("summary_file"),
            "sample_count": 10000,
            "test_accuracy": float(attempt_payload["test_accuracy"]),
            "relation_source": "online_llm",
            "llm_model": attempt_payload.get("llm_model"),
        }
        summary["comparisons"]["online_llm_minus_mock_llm"] = round(
            float(attempt_payload["test_accuracy"]) - summary["experiments"]["llm_guided"]["test_accuracy"],
            4,
        )
        summary["notes"].append("online_llm 已成功纳入同口径摘要。")
    else:
        summary["notes"].append("online_llm 已有正式尝试产物，但当前仍未成功纳入同口径结果。")

    return summary


def main() -> None:
    summary = attach_online_llm_attempt(build_summary())
    OUTPUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote same-metric summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
