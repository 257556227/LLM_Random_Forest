from __future__ import annotations

import json
import argparse
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
ATTEMPT_FILE = RESULTS_DIR / "mnist_leaf_expansion_online_llm_attempt.json"
SAME_METRIC_FILE = RESULTS_DIR / "mnist_leaf_expansion_same_metric_summary.json"
OUTPUT_FILE = RESULTS_DIR / "mnist_online_llm_repeatability_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat MNIST online LLM same-metric experiment")
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _run_once() -> dict:
    subprocess.run(
        ["python", "tools/attempt_mnist_online_llm_same_metric.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(ATTEMPT_FILE.read_text(encoding="utf-8"))


def build_repeatability_summary(repeats: int = 3) -> dict:
    same_metric_payload = json.loads(SAME_METRIC_FILE.read_text(encoding="utf-8"))
    mock_accuracy = float(same_metric_payload["experiments"]["llm_guided"]["test_accuracy"])

    runs = []
    accuracies = []
    success_count = 0
    for run_index in range(1, repeats + 1):
        payload = _run_once()
        run_row = {
            "run_index": run_index,
            "status": payload.get("status"),
            "test_accuracy": payload.get("test_accuracy"),
            "llm_model": payload.get("llm_model"),
        }
        runs.append(run_row)
        if payload.get("status") == "success" and payload.get("test_accuracy") is not None:
            success_count += 1
            accuracies.append(float(payload["test_accuracy"]))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "mnist",
        "repeats": repeats,
        "success_count": success_count,
        "mock_llm_accuracy": mock_accuracy,
        "runs": runs,
    }
    if accuracies:
        summary["online_accuracy_mean"] = float(sum(accuracies) / len(accuracies))
        summary["online_accuracy_min"] = float(min(accuracies))
        summary["online_accuracy_max"] = float(max(accuracies))
        summary["online_accuracy_std"] = float(statistics.pstdev(accuracies))
        summary["mean_minus_mock"] = round(summary["online_accuracy_mean"] - mock_accuracy, 4)
        summary["wins_over_mock_count"] = sum(1 for item in accuracies if item > mock_accuracy)
    else:
        summary["note"] = "没有成功的 online_llm 运行，无法计算稳定性统计。"
    return summary


def main() -> None:
    args = parse_args()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = build_repeatability_summary(repeats=args.repeats)
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote repeatability summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()