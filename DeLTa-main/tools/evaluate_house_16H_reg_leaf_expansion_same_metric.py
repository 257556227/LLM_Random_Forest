from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "house_16H_reg"
WITHOUT_FILE = RESULTS_DIR / "leaf_expansion_without_llm_summary.json"
MOCK_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_summary.json"
ONLINE_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
OUTPUT_FILE = RESULTS_DIR / "house_16H_reg_leaf_expansion_same_metric_summary.json"


def _read_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def build_summary() -> dict[str, Any]:
    without_payload = _read_json(WITHOUT_FILE)
    mock_payload = _read_json(MOCK_FILE)
    online_payload = _read_json(ONLINE_FILE)

    baseline_rmse = float(without_payload["baseline_rmse"])
    without_rmse = float(without_payload["expanded_rmse"])
    mock_rmse = float(mock_payload["expanded_rmse"])
    online_rmse = float(online_payload["expanded_rmse"])

    baseline_r2 = float(without_payload["baseline_r2"])
    without_r2 = float(without_payload["expanded_r2"])
    mock_r2 = float(mock_payload["expanded_r2"])
    online_r2 = float(online_payload["expanded_r2"])

    return {
        "protocol_name": "house_16H_reg_leaf_expansion_same_metric_v1",
        "dataset": "house_16H_reg",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "same_test_rmse_eval",
        "strict_same_metric_protocol_ready": False,
        "notes": [
            "这份摘要聚焦 house_16H_reg 上 leaf expansion 原型第一轮回归结果。",
            "当前 baseline 来自同一棵 depth=3 原型树，不是 DeLTa 主链 RF/TabPFN baseline。",
            "回归任务以 RMSE 降低为主指标，以 R2 提升为辅助指标。",
        ],
        "experiments": {
            "prototype_baseline": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_rmse": baseline_rmse,
                "test_r2": baseline_r2,
            },
            "without_llm": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_rmse": without_rmse,
                "test_r2": without_r2,
                "selected_leaf_ids": without_payload.get("selected_leaf_ids", []),
            },
            "mock_llm": {
                "source_file": str(MOCK_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_rmse": mock_rmse,
                "test_r2": mock_r2,
                "selected_leaf_ids": mock_payload.get("selected_leaf_ids", []),
            },
            "online_llm": {
                "source_file": str(ONLINE_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_rmse": online_rmse,
                "test_r2": online_r2,
                "selected_leaf_ids": online_payload.get("selected_leaf_ids", []),
                "llm_model": online_payload.get("llm_model"),
            },
        },
        "comparisons": {
            "without_llm_rmse_delta_vs_baseline": round(baseline_rmse - without_rmse, 4),
            "mock_llm_rmse_delta_vs_without_llm": round(without_rmse - mock_rmse, 4),
            "online_llm_rmse_delta_vs_mock": round(mock_rmse - online_rmse, 4),
            "online_llm_rmse_delta_vs_without_llm": round(without_rmse - online_rmse, 4),
            "without_llm_r2_delta_vs_baseline": round(without_r2 - baseline_r2, 4),
            "mock_llm_r2_delta_vs_without_llm": round(mock_r2 - without_r2, 4),
            "online_llm_r2_delta_vs_mock": round(online_r2 - mock_r2, 4),
            "online_llm_r2_delta_vs_without_llm": round(online_r2 - without_r2, 4),
        },
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUTPUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote house_16H_reg same-metric summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
