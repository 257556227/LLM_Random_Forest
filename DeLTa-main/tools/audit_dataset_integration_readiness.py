from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _normalize_dataset_slug(dataset: str) -> str:
    return str(dataset).replace("-", "_")


def build_dataset_readiness_report(dataset: str) -> dict:
    dataset = str(dataset)
    dataset_slug = _normalize_dataset_slug(dataset)
    dataset_dir = PROJECT_ROOT / "example_datasets" / dataset
    results_dir = PROJECT_ROOT / "results" / dataset
    prompt_file = PROJECT_ROOT / "llm" / "get_prompts" / f"{dataset}.py"
    llm_rule_file = PROJECT_ROOT / "model" / "llm_rule" / f"{dataset}.py"
    docs_precheck_file = WORKSPACE_ROOT / "docs" / f"{dataset_slug}_integration_precheck.md"
    experiment_ledger = WORKSPACE_ROOT / "experiments" / f"{dataset_slug}_experiment_ledger.md"
    module_map = PROJECT_ROOT / "docs" / f"module_map_{dataset_slug}.md"
    dataset_config = PROJECT_ROOT / "dataset_config.py"

    checks = {
        "dataset_config_exists": dataset_config.exists(),
        "prompt_template_exists": prompt_file.exists(),
        "llm_rule_exists": llm_rule_file.exists(),
        "dataset_dir_exists": dataset_dir.exists(),
        "results_dir_exists": results_dir.exists(),
        "precheck_doc_exists": docs_precheck_file.exists(),
        "experiment_ledger_exists": experiment_ledger.exists(),
        "module_map_exists": module_map.exists(),
    }

    expected_dataset_files = [
        "N_train.npy",
        "N_val.npy",
        "N_test.npy",
        "C_train.npy",
        "C_val.npy",
        "C_test.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "info.json",
    ]
    dataset_files = {name: (dataset_dir / name).exists() for name in expected_dataset_files}

    missing_items = []
    if not checks["dataset_dir_exists"]:
        missing_items.append(f"example_datasets/{dataset}")
    else:
        missing_items.extend(
            f"example_datasets/{dataset}/{name}"
            for name, exists in dataset_files.items()
            if not exists and name not in {"C_train.npy", "C_val.npy", "C_test.npy"}
        )

    if not checks["results_dir_exists"]:
        missing_items.append(f"results/{dataset}")
    if not checks["experiment_ledger_exists"]:
        missing_items.append(f"experiments/{dataset_slug}_experiment_ledger.md")
    if not checks["module_map_exists"]:
        missing_items.append(f"DeLTa-main/docs/module_map_{dataset_slug}.md")

    ready_for_first_run = (
        checks["dataset_config_exists"]
        and checks["prompt_template_exists"]
        and checks["llm_rule_exists"]
        and checks["dataset_dir_exists"]
        and dataset_files["y_train.npy"]
        and dataset_files["y_val.npy"]
        and dataset_files["y_test.npy"]
        and dataset_files["info.json"]
        and (dataset_files["N_train.npy"] or dataset_files["C_train.npy"])
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "checks": checks,
        "dataset_files": dataset_files,
        "ready_for_first_run": bool(ready_for_first_run),
        "missing_items": missing_items,
        "recommended_next_action": (
            f"先补 example_datasets/{dataset}/ 和 info.json，再启动第一轮传统底座与 leaf expansion 验证。"
            if not ready_for_first_run
            else f"可以开始 {dataset} 的第一轮统一入口验证。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit integration readiness for a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset_readiness_report(args.dataset)
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / args.dataset / f"{args.dataset}_integration_readiness.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote dataset readiness report to {output_path}")


if __name__ == "__main__":
    main()
