from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.utils import resolve_small_model_for_runtime


RESULTS_DIR = PROJECT_ROOT / "results" / "house_16H_reg"
WITHOUT_FILE = RESULTS_DIR / "leaf_expansion_without_llm_summary.json"
MOCK_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_summary.json"
ONLINE_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
MAINLINE_RF_FILE = RESULTS_DIR / "house_16H_reg_md10_ml200_tree5.npy"
MAINLINE_DELTA_FILE = RESULTS_DIR / f"RF_md10_ml200_tree5_full_{resolve_small_model_for_runtime('tabpfn')}_0.npy"
OUTPUT_FILE = RESULTS_DIR / "house_16H_reg_leaf_expansion_formal_protocol_summary.json"


def _read_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _load_dataset_loader():
    module_path = PROJECT_ROOT / "run_leaf_expansion_mnist.py"
    spec = importlib.util.spec_from_file_location("house_leaf_expansion_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_mnist_dataset


def _merge_train_val(parts: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([parts["train"], parts["val"]], axis=0)


def _load_regression_info(dataset_name: str) -> dict[str, Any]:
    file_path = PROJECT_ROOT / "reg_info" / f"{dataset_name}.json"
    return json.loads(file_path.read_text(encoding="utf-8"))


def _scaled_rmse(labels: np.ndarray, predictions: np.ndarray, scale_info: dict[str, Any]) -> float:
    rmse = float(mean_squared_error(labels, predictions) ** 0.5)
    if scale_info.get("policy") == "mean_std":
        rmse *= float(scale_info["std"])
    return rmse


def _load_mainline_rf_result(file_path: Path, scale_info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    payload = np.load(file_path, allow_pickle=True).item()
    predictions = np.asarray(payload["logit"], dtype=float)
    labels = np.asarray(payload["label"], dtype=float)
    return {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sample_count": int(labels.shape[0]),
        "test_rmse": _scaled_rmse(labels, predictions, scale_info),
        "test_r2": float(r2_score(labels, predictions)),
    }, predictions, labels


def _load_mainline_delta_result(
    file_path: Path,
    rf_predictions: np.ndarray,
    labels: np.ndarray,
    scale_info: dict[str, Any],
) -> dict[str, Any]:
    payload = np.load(file_path, allow_pickle=True).item()
    lm_predictions = np.asarray(payload["logit"], dtype=float)
    fused_predictions = lm_predictions + 0.001 * rf_predictions
    return {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sample_count": int(labels.shape[0]),
        "test_rmse": _scaled_rmse(labels, fused_predictions, scale_info),
        "test_r2": float(r2_score(labels, fused_predictions)),
        "small_model_used": resolve_small_model_for_runtime("tabpfn"),
        "fusion_eta": 0.001,
    }


def _build_deeper_rf() -> dict[str, Any]:
    load_dataset = _load_dataset_loader()
    feature_parts, target_parts, _info = load_dataset("house_16H_reg", "example_datasets")
    X_train = _merge_train_val(feature_parts)
    y_train = _merge_train_val(target_parts).astype(float)
    X_test = feature_parts["test"]
    y_test = target_parts["test"].astype(float)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "source_mode": "house_16H_reg depth=4 random_forest same-metric recompute",
        "sample_count": int(y_test.shape[0]),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "test_r2": float(r2_score(y_test, predictions)),
        "n_estimators": 100,
        "max_depth": 4,
    }


def build_summary() -> dict[str, Any]:
    without_payload = _read_json(WITHOUT_FILE)
    mock_payload = _read_json(MOCK_FILE)
    online_payload = _read_json(ONLINE_FILE)
    scale_info = _load_regression_info("house_16H_reg")

    prototype_baseline_rmse = float(without_payload["baseline_rmse"])
    prototype_baseline_r2 = float(without_payload["baseline_r2"])
    without_rmse = float(without_payload["expanded_rmse"])
    without_r2 = float(without_payload["expanded_r2"])
    mock_rmse = float(mock_payload["expanded_rmse"])
    mock_r2 = float(mock_payload["expanded_r2"])
    online_rmse = float(online_payload["expanded_rmse"])
    online_r2 = float(online_payload["expanded_r2"])

    traditional_rf, rf_predictions, labels = _load_mainline_rf_result(MAINLINE_RF_FILE, scale_info)
    traditional_delta = _load_mainline_delta_result(MAINLINE_DELTA_FILE, rf_predictions, labels, scale_info)
    deeper_rf = _build_deeper_rf()

    return {
        "protocol_name": "house_16H_reg_leaf_expansion_formal_protocol_v1",
        "dataset": "house_16H_reg",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "same_test_rmse_eval",
        "strict_same_metric_protocol_ready": True,
        "notes": [
            "这份摘要把 house_16H_reg 的 leaf expansion 原型、depth=4 Deeper RF，以及传统 RF / DeLTa 主链结果放到同一张协议表里。",
            "当前环境无 CUDA，因此传统 DeLTa 主链使用了 tabpfn -> cart 的自动运行时回退；该回退已经记录在 small_model_used 字段中。",
            "回归任务以 RMSE 降低为主指标，以 R2 提升为辅助指标。",
        ],
        "experiments": {
            "prototype_baseline": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": traditional_rf["sample_count"],
                "test_rmse": prototype_baseline_rmse,
                "test_r2": prototype_baseline_r2,
                "max_depth": 3,
            },
            "deeper_rf": deeper_rf,
            "without_llm": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": traditional_rf["sample_count"],
                "test_rmse": without_rmse,
                "test_r2": without_r2,
                "selected_leaf_ids": without_payload.get("selected_leaf_ids", []),
                "leaf_selection_strategy": without_payload.get("leaf_selection_strategy"),
            },
            "mock_llm": {
                "source_file": str(MOCK_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": traditional_rf["sample_count"],
                "test_rmse": mock_rmse,
                "test_r2": mock_r2,
                "selected_leaf_ids": mock_payload.get("selected_leaf_ids", []),
                "leaf_selection_strategy": mock_payload.get("leaf_selection_strategy"),
            },
            "online_llm": {
                "source_file": str(ONLINE_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sample_count": traditional_rf["sample_count"],
                "test_rmse": online_rmse,
                "test_r2": online_r2,
                "selected_leaf_ids": online_payload.get("selected_leaf_ids", []),
                "leaf_selection_strategy": online_payload.get("leaf_selection_strategy"),
                "llm_model": online_payload.get("llm_model"),
            },
            "traditional_rf": traditional_rf,
            "traditional_delta": traditional_delta,
        },
        "comparisons": {
            "without_llm_rmse_delta_vs_prototype_baseline": round(prototype_baseline_rmse - without_rmse, 4),
            "online_llm_rmse_delta_vs_mock": round(mock_rmse - online_rmse, 4),
            "online_llm_rmse_delta_vs_without_llm": round(without_rmse - online_rmse, 4),
            "online_llm_rmse_delta_vs_deeper_rf": round(deeper_rf["test_rmse"] - online_rmse, 4),
            "online_llm_rmse_delta_vs_traditional_rf": round(traditional_rf["test_rmse"] - online_rmse, 4),
            "online_llm_rmse_delta_vs_traditional_delta": round(traditional_delta["test_rmse"] - online_rmse, 4),
            "without_llm_r2_delta_vs_prototype_baseline": round(without_r2 - prototype_baseline_r2, 4),
            "online_llm_r2_delta_vs_mock": round(online_r2 - mock_r2, 4),
            "online_llm_r2_delta_vs_without_llm": round(online_r2 - without_r2, 4),
            "online_llm_r2_delta_vs_deeper_rf": round(online_r2 - deeper_rf["test_r2"], 4),
            "online_llm_r2_delta_vs_traditional_rf": round(online_r2 - traditional_rf["test_r2"], 4),
            "online_llm_r2_delta_vs_traditional_delta": round(online_r2 - traditional_delta["test_r2"], 4),
        },
    }


def main() -> None:
    required_files = [WITHOUT_FILE, MOCK_FILE, ONLINE_FILE, MAINLINE_RF_FILE, MAINLINE_DELTA_FILE]
    missing = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files for house_16H_reg formal protocol summary: {missing}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUTPUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote house_16H_reg formal protocol summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
