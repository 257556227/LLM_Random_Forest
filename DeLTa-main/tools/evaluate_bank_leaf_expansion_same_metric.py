from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.leaf_expansion import fit_leaf_expansion, rank_features_with_mock_llm, rank_features_without_llm, select_candidate_leaves  # noqa: E402
from run_leaf_expansion_mnist import _merge_train_val, load_mnist_dataset  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "bank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bank leaf expansion same metric summary")
    parser.add_argument("--leaf_selection_strategy", default="impurity_mass")
    parser.add_argument("--output", default="")
    parser.add_argument("--online_attempt", default="")
    return parser.parse_args()


def _default_output_file(leaf_selection_strategy: str) -> Path:
    if leaf_selection_strategy == "impurity_mass":
        return RESULTS_DIR / "bank_leaf_expansion_same_metric_summary.json"
    return RESULTS_DIR / f"bank_leaf_expansion_same_metric_{leaf_selection_strategy}.json"


def _default_online_attempt_file(leaf_selection_strategy: str) -> Path:
    if leaf_selection_strategy == "impurity_mass":
        return RESULTS_DIR / "bank_leaf_expansion_online_llm_attempt.json"
    return RESULTS_DIR / f"bank_leaf_expansion_online_llm_attempt_{leaf_selection_strategy}.json"


def _load_mainline_result(file_path: Path) -> dict[str, Any]:
    payload = np.load(file_path, allow_pickle=True).item()
    logits = np.asarray(payload["logit"])
    labels = np.asarray(payload["label"]).astype(int)
    if logits.ndim == 1:
        predictions = (logits > 0.5).astype(int)
    else:
        predictions = np.argmax(logits, axis=1)
    return {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sample_count": int(labels.shape[0]),
        "test_accuracy": float(accuracy_score(labels, predictions)),
    }


def _build_base_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    base_tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, random_state=42)
    base_tree.fit(X_train, y_train)
    return base_tree


def _leaf_expansion_same_metric(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, feature_names, feature_types, leaf_selection_strategy: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base_tree = _build_base_tree(X_train, y_train)
    leaf_ids = base_tree.apply(X_train)
    selected_contexts = select_candidate_leaves(
        base_tree,
        X_train,
        y_train,
        top_k_leaves=3,
        min_leaf_samples=80,
        top_feature_k=10,
        leaf_selection_strategy=leaf_selection_strategy,
        feature_names=feature_names,
        feature_types=feature_types,
    )

    without_ranked = {}
    llm_ranked = {}
    for context in selected_contexts:
        indices = np.where(leaf_ids == context.leaf_id)[0]
        leaf_X = X_train[indices]
        without_ranked[int(context.leaf_id)] = rank_features_without_llm(leaf_X, top_feature_ids=context.candidate_feature_ids)
        llm_ranked[int(context.leaf_id)] = rank_features_with_mock_llm(context)

    without_model = fit_leaf_expansion(base_tree, X_train, y_train, selected_contexts, without_ranked, random_state=42, top_k_features_to_try=3)
    guided_model = fit_leaf_expansion(base_tree, X_train, y_train, selected_contexts, llm_ranked, random_state=42, top_k_features_to_try=3)

    without_summary = {
        "source_mode": "run_leaf_expansion bank without_llm equivalent",
        "sample_count": int(y_test.shape[0]),
        "test_accuracy": float(accuracy_score(y_test, without_model.predict(X_test))),
        "selected_leaf_ids": [int(context.leaf_id) for context in selected_contexts],
        "leaf_selection_strategy": leaf_selection_strategy,
    }
    guided_summary = {
        "source_mode": "run_leaf_expansion bank llm_guided mock_llm equivalent",
        "sample_count": int(y_test.shape[0]),
        "test_accuracy": float(accuracy_score(y_test, guided_model.predict(X_test))),
        "selected_leaf_ids": [int(context.leaf_id) for context in selected_contexts],
        "relation_source": "mock_llm",
        "leaf_selection_strategy": leaf_selection_strategy,
    }
    return without_summary, guided_summary


def build_summary(leaf_selection_strategy: str) -> dict[str, Any]:
    feature_parts, target_parts, info = load_mnist_dataset("bank", "example_datasets")
    X_train = _merge_train_val(feature_parts)
    y_train = _merge_train_val(target_parts).astype(int)
    X_test = feature_parts["test"]
    y_test = target_parts["test"].astype(int)

    baseline_rf = _load_mainline_result(RESULTS_DIR / "bank_md20_ml50_tree15.npy")
    without_llm, llm_guided = _leaf_expansion_same_metric(
        X_train,
        y_train,
        X_test,
        y_test,
        info.get("leaf_expansion_feature_names"),
        info.get("leaf_expansion_feature_types"),
        leaf_selection_strategy,
    )

    return {
        "protocol_name": f"bank_leaf_expansion_same_metric_{leaf_selection_strategy}_v1",
        "dataset": "bank",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "same_test_accuracy_eval",
        "strict_same_metric_protocol_ready": True,
        "leaf_selection_strategy": leaf_selection_strategy,
        "experiments": {
            "baseline_rf": baseline_rf,
            "without_llm": without_llm,
            "llm_guided": llm_guided,
        },
        "comparisons": {
            "llm_guided_minus_without": round(llm_guided["test_accuracy"] - without_llm["test_accuracy"], 4),
            "llm_guided_minus_baseline": round(llm_guided["test_accuracy"] - baseline_rf["test_accuracy"], 4),
        },
        "notes": [
            "这份摘要比较 bank 上 leaf expansion 原型的统一 test accuracy。",
            "baseline_rf 来自主链 .npy；without_llm 与 llm_guided(mock_llm) 由同一份 bank 数据和固定原型参数现场重算。",
        ],
    }


def attach_online_llm_attempt(summary: dict[str, Any], online_attempt_file: Path) -> dict[str, Any]:
    if not online_attempt_file.exists():
        return summary

    attempt_payload = json.loads(online_attempt_file.read_text(encoding="utf-8"))
    summary["online_llm_attempt"] = attempt_payload
    if attempt_payload.get("status") == "success" and attempt_payload.get("test_accuracy") is not None:
        summary["experiments"]["online_llm"] = {
            "source_file": attempt_payload.get("summary_file"),
            "sample_count": 9043,
            "test_accuracy": float(attempt_payload["test_accuracy"]),
            "relation_source": "online_llm",
            "llm_model": attempt_payload.get("llm_model"),
        }
        summary["comparisons"]["online_llm_minus_mock_llm"] = round(
            float(attempt_payload["test_accuracy"]) - summary["experiments"]["llm_guided"]["test_accuracy"],
            4,
        )
        summary["notes"].append("bank online_llm 已成功纳入同口径摘要。")
    else:
        summary["notes"].append("bank online_llm 已有正式尝试产物，但当前仍未成功纳入同口径结果。")
    return summary


def main() -> None:
    args = parse_args()
    output_file = Path(args.output) if args.output else _default_output_file(args.leaf_selection_strategy)
    online_attempt_file = Path(args.online_attempt) if args.online_attempt else _default_online_attempt_file(args.leaf_selection_strategy)
    summary = attach_online_llm_attempt(build_summary(args.leaf_selection_strategy), online_attempt_file)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bank same-metric summary to {output_file}")


if __name__ == "__main__":
    main()