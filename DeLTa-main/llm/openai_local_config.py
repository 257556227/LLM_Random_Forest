from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_FILE_NAME = "local_openai_config.json"


def _candidate_paths() -> list[Path]:
    return [
        Path.cwd() / CONFIG_FILE_NAME,
        PROJECT_ROOT / CONFIG_FILE_NAME,
        WORKSPACE_ROOT / CONFIG_FILE_NAME,
    ]


def find_local_openai_config() -> Path | None:
    for path in _candidate_paths():
        if path.exists():
            return path
    return None


def load_local_openai_config() -> tuple[dict[str, Any], Path | None]:
    config_path = find_local_openai_config()
    if config_path is None:
        return {}, None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, config_path

    if not isinstance(payload, dict):
        return {}, config_path
    return payload, config_path


def _pick_non_empty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return default


def get_openai_settings(explicit_model: str | None = None) -> dict[str, Any]:
    local_config, config_path = load_local_openai_config()

    api_key = _pick_non_empty(
        local_config.get("OPENAI_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    )
    base_url = _pick_non_empty(
        local_config.get("OPENAI_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        default=DEFAULT_OPENAI_BASE_URL,
    )
    model = _pick_non_empty(
        explicit_model,
        local_config.get("OPENAI_MODEL"),
        os.environ.get("OPENAI_MODEL"),
        default=DEFAULT_OPENAI_MODEL,
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "config_path": str(config_path) if config_path else "",
        "has_api_key": bool(api_key),
        "has_base_url": bool(base_url),
        "has_model": bool(model),
    }
