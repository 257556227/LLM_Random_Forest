from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_paths(script_path: Path) -> dict[str, Path]:
    delta_root = script_path.parents[1]
    results_root = delta_root / "results" / "mnist"
    return {
        "results_root": results_root,
        "baseline_log": results_root / "e_RF_md8_ml20_tree3_mnist_cart_9.log",
        "deeper_log": results_root / "e_RF_md20_ml100_tree15_mnist_cart_9.log",
        "without_llm_summary": results_root / "leaf_expansion_without_llm_summary.json",
        "llm_guided_summary": results_root / "llm_guided_leaf_expansion_summary.json",
        "output": results_root / "mnist_leaf_expansion_protocol_summary.json",
    }


def _extract_float(log_text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)", log_text)
    if not match:
        raise ValueError(f"未在日志中找到指标: {label}")
    return float(match.group(1))


def _extract_optional_float(log_text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)", log_text)
    if not match:
        return None
    return float(match.group(1))


def _load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _parse_mainline_log(log_path: Path, label: str, config: dict[str, Any]) -> dict[str, Any]:
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "label": label,
        "metric_scope": "delta_mainline_fused_eval",
        "source_file": str(log_path.relative_to(log_path.parents[2])).replace("\\", "/"),
        "config": config,
        "rf_accuracy_mean": _extract_float(log_text, "RF Accuracy MEAN"),
        "rf_auc_mean": _extract_optional_float(log_text, "RF AUC MEAN"),
        "fused_accuracy_mean": _extract_float(log_text, "Fused Accuracy MEAN"),
        "fused_auc_mean": _extract_optional_float(log_text, "Fused AUC MEAN"),
        "eta": _extract_optional_float(log_text, "Fusion eta"),
    }


def _parse_leaf_summary(summary_path: Path, label: str) -> dict[str, Any]:
    summary = _load_json(summary_path)
    return {
        "label": label,
        "metric_scope": "leaf_expansion_single_model_eval",
        "source_file": str(summary_path.relative_to(summary_path.parents[2])).replace("\\", "/"),
        "prototype": summary.get("prototype"),
        "baseline_accuracy": summary.get("baseline_accuracy"),
        "expanded_accuracy": summary.get("expanded_accuracy"),
        "selected_leaf_ids": summary.get("selected_leaf_ids", []),
        "expanded_leaf_count": summary.get("expanded_leaf_count"),
        "relation_source": summary.get("relation_source", "without_llm"),
        "dataset": summary.get("dataset"),
        "task_type": summary.get("task_type"),
    }


def build_protocol_summary(paths: dict[str, Path]) -> dict[str, Any]:
    baseline = _parse_mainline_log(
        paths["baseline_log"],
        label="Baseline RF",
        config={"md": 8, "ml": 20, "tree": 3, "n_ensemble": 9, "eta": 0.2},
    )
    deeper = _parse_mainline_log(
        paths["deeper_log"],
        label="Deeper RF",
        config={"md": 20, "ml": 100, "tree": 15, "n_ensemble": 9},
    )
    without_llm = _parse_leaf_summary(paths["without_llm_summary"], label="Leaf Expansion without LLM")
    llm_guided = _parse_leaf_summary(paths["llm_guided_summary"], label="LLM-Guided Leaf Expansion")

    return {
        "protocol_name": "mnist_leaf_expansion_four_way_v1",
        "dataset": "mnist",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_same_metric_protocol_ready": False,
        "protocol_status": "artifact_inventory_ready_but_not_same_metric_rerun",
        "experiments": {
            "baseline_rf": baseline,
            "deeper_rf": deeper,
            "without_llm": without_llm,
            "llm_guided": llm_guided,
        },
        "comparisons": {
            "delta_mainline_fused_gain_deeper_minus_baseline": round(
                deeper["fused_accuracy_mean"] - baseline["fused_accuracy_mean"], 4
            ),
            "leaf_expansion_gain_without_llm": round(
                without_llm["expanded_accuracy"] - without_llm["baseline_accuracy"], 4
            ),
            "leaf_expansion_gain_llm_guided": round(
                llm_guided["expanded_accuracy"] - llm_guided["baseline_accuracy"], 4
            ),
            "leaf_expansion_extra_gain_llm_minus_without": round(
                llm_guided["expanded_accuracy"] - without_llm["expanded_accuracy"], 4
            ),
        },
        "notes": [
            "当前四组协议已经有统一汇总产物，但还不是严格同口径复跑后的最终协议。",
            "Baseline RF 与 Deeper RF 来自 DeLTa 主链的 fused eval；without_llm 与 llm_guided 来自叶子扩展原型的 single-model eval。",
            "因此这份摘要当前用于阶段一收口与产物盘点，不应用来直接声明最终 apples-to-apples 结论。",
            "下一步应该把四组实验迁移到同一评估协议下重跑，再决定是否把 strict_same_metric_protocol_ready 改为 true。",
        ],
    }


def main() -> None:
    script_path = Path(__file__).resolve()
    paths = _default_paths(script_path)
    summary = build_protocol_summary(paths)
    paths["output"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote summary to {paths['output']}")


if __name__ == "__main__":
    main()
