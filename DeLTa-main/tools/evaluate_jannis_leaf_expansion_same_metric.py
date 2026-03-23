from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "jannis"
WITHOUT_FILE = RESULTS_DIR / "leaf_expansion_without_llm_summary.json"
MOCK_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_summary.json"
ONLINE_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
OUTPUT_FILE = RESULTS_DIR / "jannis_leaf_expansion_same_metric_summary.json"


def _read_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def build_summary() -> dict[str, Any]:
    without_payload = _read_json(WITHOUT_FILE)
    mock_payload = _read_json(MOCK_FILE)
    online_payload = _read_json(ONLINE_FILE)

    baseline_accuracy = float(without_payload["baseline_accuracy"])
    without_accuracy = float(without_payload["expanded_accuracy"])
    mock_accuracy = float(mock_payload["expanded_accuracy"])
    online_accuracy = float(online_payload["expanded_accuracy"])

    return {
        "protocol_name": "jannis_leaf_expansion_same_metric_v1",
        "dataset": "jannis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "same_test_accuracy_eval",
        "strict_same_metric_protocol_ready": False,
        "notes": [
            "这份摘要聚焦 jannis 上 leaf expansion 原型第一轮结果。",
            "当前 baseline 来自同一棵 depth=3 原型树，不是 DeLTa 主链 RF baseline。",
            "jannis 属于高维弱语义多分类数据，第一轮重点看 online 是否能超过 mock 与 without_llm。",
        ],
        "experiments": {
            "prototype_baseline": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_accuracy": baseline_accuracy,
            },
            "without_llm": {
                "source_file": str(WITHOUT_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_accuracy": without_accuracy,
                "selected_leaf_ids": without_payload.get("selected_leaf_ids", []),
            },
            "mock_llm": {
                "source_file": str(MOCK_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_accuracy": mock_accuracy,
                "selected_leaf_ids": mock_payload.get("selected_leaf_ids", []),
            },
            "online_llm": {
                "source_file": str(ONLINE_FILE.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                "test_accuracy": online_accuracy,
                "selected_leaf_ids": online_payload.get("selected_leaf_ids", []),
                "llm_model": online_payload.get("llm_model"),
            },
        },
        "comparisons": {
            "without_llm_minus_baseline": round(without_accuracy - baseline_accuracy, 4),
            "mock_llm_minus_without_llm": round(mock_accuracy - without_accuracy, 4),
            "online_llm_minus_mock_llm": round(online_accuracy - mock_accuracy, 4),
            "online_llm_minus_without_llm": round(online_accuracy - without_accuracy, 4),
        },
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUTPUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote jannis same-metric summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()