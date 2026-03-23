from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _merge_train_val(parts: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([parts["train"], parts["val"]], axis=0)


def _load_dataset_loader():
    module_path = PROJECT_ROOT / "run_leaf_expansion_mnist.py"
    spec = importlib.util.spec_from_file_location("credit_g_leaf_expansion_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_mnist_dataset


RESULTS_DIR = PROJECT_ROOT / "results" / "credit-g"
WITHOUT_FILE = RESULTS_DIR / "leaf_expansion_without_llm_summary.json"
MOCK_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_summary.json"
ONLINE_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
MAINLINE_RF_FILE = RESULTS_DIR / "credit-g_md15_ml20_tree20.npy"
MAINLINE_DELTA_LOG = RESULTS_DIR / "e_RF_md15_ml20_tree20_credit-g_cart_0.log"
OUTPUT_FILE = RESULTS_DIR / "credit_g_leaf_expansion_formal_protocol_summary.json"


def _read_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _load_mainline_rf_result(file_path: Path) -> dict[str, Any]:
    payload = np.load(file_path, allow_pickle=True).item()
    logits = np.asarray(payload["logit"])
    labels = np.asarray(payload["label"]).astype(int)
    predictions = np.argmax(logits, axis=1)
    positive_scores = logits[:, 1] if logits.ndim == 2 and logits.shape[1] > 1 else predictions
    return {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sample_count": int(labels.shape[0]),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_auc": float(roc_auc_score(labels, positive_scores)),
    }


def _parse_classification_metric(log_text: str, metric_name: str) -> float:
    for raw_line in log_text.splitlines():
        normalized_line = raw_line.strip().replace("\x00", "")
        if metric_name not in normalized_line:
            continue
        try:
            return float(normalized_line.split(":", 1)[1].strip())
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Failed to parse '{metric_name}' from line: {raw_line!r}") from exc
    raise ValueError(f"Could not find '{metric_name}' in ensemble log")


def _load_mainline_delta_result(log_file: Path) -> dict[str, Any]:
    log_text = log_file.read_text(encoding="utf-8", errors="ignore")
    return {
        "source_file": str(log_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sample_count": None,
        "test_accuracy": _parse_classification_metric(log_text, "Fused Accuracy MEAN"),
        "test_auc": _parse_classification_metric(log_text, "Fused AUC MEAN"),
        "eta": _parse_classification_metric(log_text, "Fusion eta"),
    }


def _build_deeper_rf(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    predictions = np.argmax(probabilities, axis=1)
    return {
        "source_mode": "credit-g depth=4 random_forest same-metric recompute",
        "sample_count": int(y_test.shape[0]),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_auc": float(roc_auc_score(y_test, probabilities[:, 1])),
        "n_estimators": 100,
        "max_depth": 4,
    }


def build_summary() -> dict[str, Any]:
    without_payload = _read_json(WITHOUT_FILE)
    mock_payload = _read_json(MOCK_FILE)
    online_payload = _read_json(ONLINE_FILE)

    load_dataset = _load_dataset_loader()
    feature_parts, target_parts, _info = load_dataset("credit-g", "example_datasets")
    X_train = _merge_train_val(feature_parts)
    y_train = _merge_train_val(target_parts).astype(int)
    X_test = feature_parts["test"]
    y_test = target_parts["test"].astype(int)

    prototype_baseline_accuracy = float(without_payload["baseline_accuracy"])
    without_accuracy = float(without_payload["expanded_accuracy"])
    mock_accuracy = float(mock_payload["expanded_accuracy"])
    online_accuracy = float(online_payload["expanded_accuracy"])

    traditional_rf = _load_mainline_rf_result(MAINLINE_RF_FILE)
    traditional_delta = _load_mainline_delta_result(MAINLINE_DELTA_LOG)
    deeper_rf = _build_deeper_rf(X_train, y_train, X_test, y_test)

    return {
        "protocol_name": "credit_g_leaf_expansion_formal_protocol_v1",
        "dataset": "credit-g",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "same_test_accuracy_eval",
        "strict_same_metric_protocol_ready": True,
        "experiments": {
            "prototype_baseline": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": int(y_test.shape[0]),
                "test_accuracy": prototype_baseline_accuracy,
                "max_depth": 3,
            },
            "deeper_rf": deeper_rf,
            "without_llm": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": int(y_test.shape[0]),
                "test_accuracy": without_accuracy,
                "selected_leaf_ids": without_payload.get("selected_leaf_ids", []),
                "leaf_selection_strategy": without_payload.get("leaf_selection_strategy"),
            },
            "mock_llm": {
                "source_file": str(MOCK_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": int(y_test.shape[0]),
                "test_accuracy": mock_accuracy,
                "selected_leaf_ids": mock_payload.get("selected_leaf_ids", []),
                "leaf_selection_strategy": mock_payload.get("leaf_selection_strategy"),
            },
            "online_llm": {
                "source_file": str(ONLINE_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": int(y_test.shape[0]),
                "test_accuracy": online_accuracy,
                "selected_leaf_ids": online_payload.get("selected_leaf_ids", []),
                "leaf_selection_strategy": online_payload.get("leaf_selection_strategy"),
                "llm_model": online_payload.get("llm_model"),
            },
            "traditional_rf": traditional_rf,
            "traditional_delta": traditional_delta,
        },
        "comparisons": {
            "without_llm_minus_prototype_baseline": round(without_accuracy - prototype_baseline_accuracy, 4),
            "online_llm_minus_mock_llm": round(online_accuracy - mock_accuracy, 4),
            "online_llm_minus_without_llm": round(online_accuracy - without_accuracy, 4),
            "online_llm_minus_deeper_rf": round(online_accuracy - deeper_rf["test_accuracy"], 4),
            "online_llm_minus_traditional_rf": round(online_accuracy - traditional_rf["test_accuracy"], 4),
            "online_llm_minus_traditional_delta": round(online_accuracy - traditional_delta["test_accuracy"], 4),
        },
        "notes": [
            "这份摘要把 credit-g 的 leaf expansion 原型、depth=4 Deeper RF，以及传统 DeLTa/RF 主链结果放到同一张协议表里。",
            "deeper_rf 使用相同 credit-g 数据划分现场重算的 depth=4 RandomForestClassifier，用来对应统一协议中的 Deeper RF 主对照。",
            "traditional_rf 来自 run_randforest.py 的主链 .npy；traditional_delta 来自 train.py + ensemble.py 的融合日志。",
        ],
    }


def main() -> None:
    required_files = [WITHOUT_FILE, MOCK_FILE, ONLINE_FILE, MAINLINE_RF_FILE, MAINLINE_DELTA_LOG]
    missing = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files for credit-g formal protocol summary: {missing}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUTPUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote credit-g formal protocol summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()