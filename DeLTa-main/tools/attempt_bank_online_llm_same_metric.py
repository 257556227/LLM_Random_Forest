from __future__ import annotations

import json
import importlib.util
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_openai_settings_helper():
    helper_path = PROJECT_ROOT / "llm" / "openai_local_config.py"
    spec = importlib.util.spec_from_file_location("delta_openai_local_config", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OpenAI config helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_openai_settings


get_openai_settings = _load_openai_settings_helper()
RESULTS_DIR = PROJECT_ROOT / "results" / "bank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attempt bank online_llm same-metric run")
    parser.add_argument("--leaf_selection_strategy", default="impurity_mass")
    parser.add_argument("--guided_output", default="")
    parser.add_argument("--attempt_output", default="")
    return parser.parse_args()


def _default_summary_file(leaf_selection_strategy: str) -> Path:
    if leaf_selection_strategy == "impurity_mass":
        return RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
    return RESULTS_DIR / f"llm_guided_leaf_expansion_online_{leaf_selection_strategy}_summary.json"


def _default_attempt_file(leaf_selection_strategy: str) -> Path:
    if leaf_selection_strategy == "impurity_mass":
        return RESULTS_DIR / "bank_leaf_expansion_online_llm_attempt.json"
    return RESULTS_DIR / f"bank_leaf_expansion_online_llm_attempt_{leaf_selection_strategy}.json"


def _classify_status(error_message: str) -> str:
    normalized = error_message.lower()
    if "openai_api_key" in normalized or "required" in normalized:
        return "blocked"
    if "cloudflare" in normalized or "403" in normalized or "connection" in normalized or "timeout" in normalized:
        return "blocked"
    return "failed"


def build_blocked_payload(error_message: str, online_summary_file: Path) -> dict[str, Any]:
    openai_settings = get_openai_settings()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "bank",
        "relation_source": "online_llm",
        "status": _classify_status(error_message),
        "summary_file": str(online_summary_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "has_openai_api_key": openai_settings["has_api_key"],
        "has_openai_base_url": openai_settings["has_base_url"],
        "has_openai_model": openai_settings["has_model"],
        "config_path": openai_settings["config_path"],
        "error_message": error_message,
    }


def build_success_payload(summary: dict[str, Any], online_summary_file: Path) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "bank",
        "relation_source": "online_llm",
        "status": "success",
        "summary_file": str(online_summary_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "baseline_accuracy": summary.get("baseline_accuracy"),
        "test_accuracy": summary.get("expanded_accuracy"),
        "llm_model": summary.get("llm_model"),
        "leaf_selection_strategy": summary.get("leaf_selection_strategy", "impurity_mass"),
    }


def attempt_online_llm(leaf_selection_strategy: str, online_summary_file: Path) -> dict[str, Any]:
    command = [
        "python",
        "run_leaf_expansion_mnist.py",
        "--dataset",
        "bank",
        "--mode",
        "llm_guided",
        "--relation_source",
        "online_llm",
        "--base_max_depth",
        "3",
        "--base_max_leaf_nodes",
        "20",
        "--top_k_leaves",
        "3",
        "--min_leaf_samples",
        "80",
        "--top_feature_k",
        "10",
        "--leaf_selection_strategy",
        leaf_selection_strategy,
        "--top_k_features_to_try",
        "3",
        "--save_summary",
        "--guided_output",
        str(online_summary_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "--output",
        "results/bank/leaf_expansion_without_llm_summary.json",
    ]

    openai_settings = get_openai_settings()
    if openai_settings["model"]:
        command.extend(["--llm_model", openai_settings["model"]])

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        error_message = (exc.stderr or exc.stdout or str(exc)).strip()
        return build_blocked_payload(error_message, online_summary_file)
    except OSError as exc:
        return build_blocked_payload(str(exc), online_summary_file)

    if not online_summary_file.exists():
        return build_blocked_payload("bank online_llm run finished without creating summary file", online_summary_file)

    summary = json.loads(online_summary_file.read_text(encoding="utf-8"))
    payload = build_success_payload(summary, online_summary_file)
    payload["stdout_preview"] = (completed.stdout or "")[-1000:]
    return payload


def main() -> None:
    args = parse_args()
    online_summary_file = Path(args.guided_output) if args.guided_output else _default_summary_file(args.leaf_selection_strategy)
    attempt_file = Path(args.attempt_output) if args.attempt_output else _default_attempt_file(args.leaf_selection_strategy)
    attempt_file.parent.mkdir(parents=True, exist_ok=True)
    payload = attempt_online_llm(args.leaf_selection_strategy, online_summary_file)
    attempt_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bank online LLM attempt status to {attempt_file}")


if __name__ == "__main__":
    main()