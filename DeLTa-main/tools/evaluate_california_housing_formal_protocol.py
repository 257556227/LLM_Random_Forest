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

RESULTS_DIR = PROJECT_ROOT / "results" / "california_housing"
WITHOUT_FILE = RESULTS_DIR / "leaf_expansion_without_llm_summary.json"
MOCK_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_summary.json"
ONLINE_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
MAINLINE_RF_FILE = RESULTS_DIR / "california_housing_md10_ml200_tree5.npy"
MAINLINE_DELTA_FILE = RESULTS_DIR / f"RF_md10_ml200_tree5_full_{resolve_small_model_for_runtime('tabpfn')}_0.npy"
OUTPUT_FILE = RESULTS_DIR / "california_housing_leaf_expansion_formal_protocol_summary.json"


def _read_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _load_dataset_loader():
    module_path = PROJECT_ROOT / "run_leaf_expansion_mnist.py"
    spec = importlib.util.spec_from_file_location("california_leaf_expansion_runner", module_path)
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
    feature_parts, target_parts, _info = load_dataset("california_housing", "example_datasets")
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
        "source_mode": "california_housing depth=4 random_forest same-metric recompute",
        "sample_count": int(y_test.shape[0]),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "test_r2": float(r2_score(y_test, predictions)),
        "n_estimators": 100,
        "max_depth": 4,
    }


def build_summary() -> dict[str, Any]:
    summary = {
        "dataset": "california_housing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mainline_rf": None,
        "mainline_delta": None,
        "deeper_rf": None,
        "without_llm": None,
        "mock_llm": None,
        "online_llm": None,
    }
    scale_info = _load_regression_info("california_housing")
    # 主链路结果
    if MAINLINE_RF_FILE.exists():
        mainline_rf, rf_preds, labels = _load_mainline_rf_result(MAINLINE_RF_FILE, scale_info)
        summary["mainline_rf"] = mainline_rf
        if MAINLINE_DELTA_FILE.exists():
            summary["mainline_delta"] = _load_mainline_delta_result(MAINLINE_DELTA_FILE, rf_preds, labels, scale_info)
    # Deeper RF
    try:
        summary["deeper_rf"] = _build_deeper_rf()
    except Exception as e:
        summary["deeper_rf"] = {"error": str(e)}
    # LLM leaf expansion
    if WITHOUT_FILE.exists():
        summary["without_llm"] = _read_json(WITHOUT_FILE)
    if MOCK_FILE.exists():
        summary["mock_llm"] = _read_json(MOCK_FILE)
    if ONLINE_FILE.exists():
        summary["online_llm"] = _read_json(ONLINE_FILE)
    return summary


def main():
    summary = build_summary()
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[california_housing] formal protocol summary written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
