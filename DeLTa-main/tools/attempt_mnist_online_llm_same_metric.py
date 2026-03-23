from __future__ import annotations

import json
import importlib.util
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
RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
ONLINE_SUMMARY_FILE = RESULTS_DIR / "llm_guided_leaf_expansion_online_summary.json"
ATTEMPT_FILE = RESULTS_DIR / "mnist_leaf_expansion_online_llm_attempt.json"


def _classify_status(error_message: str) -> str:
    normalized = error_message.lower()
    if "openai_api_key" in normalized or "required" in normalized:
        return "blocked"
    if "cloudflare" in normalized or "403" in normalized or "connection" in normalized or "timeout" in normalized:
        return "blocked"
    return "failed"


def build_blocked_payload(error_message: str) -> dict[str, Any]:
    openai_settings = get_openai_settings()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "relation_source": "online_llm",
        "status": _classify_status(error_message),
        "summary_file": str(ONLINE_SUMMARY_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "has_openai_api_key": openai_settings["has_api_key"],
        "has_openai_base_url": openai_settings["has_base_url"],
        "has_openai_model": openai_settings["has_model"],
        "config_path": openai_settings["config_path"],
        "error_message": error_message,
    }


def build_success_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "relation_source": "online_llm",
        "status": "success",
        "summary_file": str(ONLINE_SUMMARY_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "baseline_accuracy": summary.get("baseline_accuracy"),
        "test_accuracy": summary.get("expanded_accuracy"),
        "llm_model": summary.get("llm_model"),
    }


def attempt_online_llm() -> dict[str, Any]:
    command = [
        "python",
        "run_leaf_expansion_mnist.py",
        "--mode",
        "llm_guided",
        "--relation_source",
        "online_llm",
        "--save_summary",
        "--guided_output",
        str(ONLINE_SUMMARY_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
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
        return build_blocked_payload(error_message)
    except OSError as exc:
        return build_blocked_payload(str(exc))

    if not ONLINE_SUMMARY_FILE.exists():
        return build_blocked_payload("online_llm run finished without creating summary file")

    summary = json.loads(ONLINE_SUMMARY_FILE.read_text(encoding="utf-8"))
    payload = build_success_payload(summary)
    payload["stdout_preview"] = (completed.stdout or "")[-1000:]
    return payload


def main() -> None:
    ATTEMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = attempt_online_llm()
    ATTEMPT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote online LLM attempt status to {ATTEMPT_FILE}")


if __name__ == "__main__":
    main()
