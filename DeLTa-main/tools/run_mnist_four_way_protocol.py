from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or preview the MNIST four-way protocol")
    parser.add_argument(
        "--config",
        default="configs/leaf_expansion/mnist_four_way_protocol.json",
        help="Path to the protocol config JSON, relative to DeLTa-main or absolute path.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Optional stage name filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the resolved commands without executing them.",
    )
    parser.add_argument(
        "--json_output",
        default="",
        help="Optional path to write the resolved execution plan as JSON.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> tuple[Path, dict[str, Any]]:
    raw_path = Path(config_path)
    if raw_path.is_absolute():
        resolved_path = raw_path
    else:
        resolved_path = Path(__file__).resolve().parents[1] / raw_path
    with resolved_path.open("r", encoding="utf-8") as file:
        return resolved_path, json.load(file)


def resolve_stages(config: dict[str, Any], selected_names: list[str]) -> list[dict[str, Any]]:
    stages = config.get("stages", [])
    if not selected_names:
        return stages
    selected_name_set = set(selected_names)
    resolved = [stage for stage in stages if stage.get("name") in selected_name_set]
    missing = selected_name_set.difference({stage.get("name") for stage in resolved})
    if missing:
        raise ValueError(f"Unknown stage(s): {', '.join(sorted(missing))}")
    return resolved


def build_plan(config_path: Path, config: dict[str, Any], selected_stages: list[dict[str, Any]]) -> dict[str, Any]:
    delta_root = config_path.parents[2]
    return {
        "protocol_name": config.get("protocol_name"),
        "dataset": config.get("dataset"),
        "config_file": str(config_path.relative_to(delta_root)).replace("\\", "/"),
        "selected_stage_names": [stage.get("name") for stage in selected_stages],
        "stages": selected_stages,
    }


def print_plan(plan: dict[str, Any]) -> None:
    print(f"Protocol: {plan['protocol_name']} ({plan['dataset']})")
    for stage in plan["stages"]:
        print(f"[{stage['name']}] kind={stage['kind']} :: {stage['description']}")
        for command in stage.get("commands", []):
            print(f"  {command}")


def execute_stage_commands(stage: dict[str, Any], working_dir: Path) -> None:
    for command in stage.get("commands", []):
        print(f"Executing: {command}")
        subprocess.run(command, cwd=working_dir, check=True, shell=True)


def main() -> None:
    args = parse_args()
    config_path, config = load_config(args.config)
    selected_stages = resolve_stages(config, args.stage)
    plan = build_plan(config_path, config, selected_stages)

    if args.json_output:
        json_output_path = Path(args.json_output)
        if not json_output_path.is_absolute():
            json_output_path = Path(__file__).resolve().parents[1] / json_output_path
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print_plan(plan)
    if args.dry_run:
        return

    working_dir = config_path.parents[2]
    for stage in selected_stages:
        execute_stage_commands(stage, working_dir)


if __name__ == "__main__":
    main()
