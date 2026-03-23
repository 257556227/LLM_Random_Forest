from __future__ import annotations

import json
import importlib.util
import re
import time
from pathlib import Path

import numpy as np
from openai import APIConnectionError, APIError, BadRequestError, OpenAI, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_openai_settings_helper():
    helper_path = PROJECT_ROOT / "llm" / "openai_local_config.py"
    spec = importlib.util.spec_from_file_location("delta_openai_local_config", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OpenAI config helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_openai_settings

get_openai_settings = _load_openai_settings_helper()


def rank_features_without_llm(leaf_X: np.ndarray, top_feature_ids: list[int] | None = None) -> list[int]:
    if leaf_X.ndim != 2:
        raise ValueError("leaf_X must be a 2D array")

    if top_feature_ids is None or len(top_feature_ids) == 0:
        candidate_ids = np.arange(leaf_X.shape[1])
    else:
        candidate_ids = np.array(top_feature_ids, dtype=int)

    candidate_matrix = leaf_X[:, candidate_ids]
    variances = np.var(candidate_matrix, axis=0)
    sorted_indices = np.argsort(-variances)
    ranked = candidate_ids[sorted_indices]
    return [int(feature_id) for feature_id in ranked.tolist()]


def build_llm_leaf_prompt(context) -> str:
    prompt_metadata = getattr(context, "prompt_metadata", {}) or {}
    task_type = str(prompt_metadata.get("task_type", ""))
    is_regression = task_type == "regression"
    distribution_text = ", ".join(
        f"class {label}: {count}" for label, count in sorted(context.class_distribution.items())
    )
    target_summary = getattr(context, "target_summary", {}) or {}
    target_summary_text = ""
    if target_summary:
        target_summary_text = ", ".join(
            f"{key}: {float(value):.6f}" for key, value in target_summary.items()
        )
    candidate_text = ", ".join(str(feature_id) for feature_id in context.candidate_feature_ids)
    candidate_summaries = getattr(context, "candidate_feature_summaries", [])
    feature_types = {
        str(item.get("feature_type", "numeric"))
        for item in candidate_summaries
    }
    feature_names = [str(item.get("feature_name", item.get("feature_id", "unknown"))) for item in candidate_summaries]
    is_tabular_context = ("categorical" in feature_types) or any(not name.isdigit() for name in feature_names)
    candidate_summary_lines = []
    for item in candidate_summaries:
        pixel_row = item.get("pixel_row", -1)
        pixel_col = item.get("pixel_col", -1)
        feature_name = str(item.get("feature_name", item.get("feature_id", "unknown")))
        feature_description = str(item.get("feature_description", feature_name))
        feature_type = str(item.get("feature_type", "numeric"))
        category_values_preview = [str(value) for value in item.get("category_values_preview", [])]
        predicted_class_top_categories = [str(value) for value in item.get("predicted_class_top_categories", [])]
        other_class_top_categories = [str(value) for value in item.get("other_class_top_categories", [])]
        pixel_text = ""
        if pixel_row is not None and pixel_col is not None and pixel_row >= 0 and pixel_col >= 0:
            pixel_text = f", pixel=({int(pixel_row)},{int(pixel_col)})"
        category_text = ""
        if category_values_preview:
            category_text += f", category_values_preview={category_values_preview}"
        if predicted_class_top_categories:
            category_text += f", predicted_class_top_categories={predicted_class_top_categories}"
        if other_class_top_categories:
            category_text += f", other_class_top_categories={other_class_top_categories}"
        if is_regression:
            candidate_summary_lines.append(
                "feature_id={feature_id}, feature_name={feature_name}, feature_description={feature_description}, feature_type={feature_type}{pixel_text}{category_text}, variance={variance:.6f}, variance_reduction={variance_reduction:.6f}, one_feature_train_rmse={one_feature_train_rmse:.6f}, target_mean_spread={target_mean_spread:.6f}, predicted_value_gap={predicted_value_gap:.6f}, predicted_value_mean={predicted_value_mean:.6f}, other_value_mean={other_value_mean:.6f}".format(
                    feature_id=int(item["feature_id"]),
                    feature_name=feature_name,
                    feature_description=feature_description,
                    feature_type=feature_type,
                    pixel_text=pixel_text,
                    category_text=category_text,
                    variance=float(item.get("variance", 0.0)),
                    variance_reduction=float(item.get("variance_reduction", 0.0)),
                    one_feature_train_rmse=float(item.get("one_feature_train_rmse", 0.0)),
                    target_mean_spread=float(item.get("target_mean_spread", 0.0)),
                    predicted_value_gap=float(item.get("predicted_value_gap", 0.0)),
                    predicted_value_mean=float(item.get("predicted_value_mean", 0.0)),
                    other_value_mean=float(item.get("other_value_mean", 0.0)),
                )
            )
        else:
            candidate_summary_lines.append(
                "feature_id={feature_id}, feature_name={feature_name}, feature_description={feature_description}, feature_type={feature_type}{pixel_text}{category_text}, variance={variance:.6f}, information_gain={information_gain:.6f}, one_feature_train_accuracy={one_feature_train_accuracy:.6f}, class_mean_spread={class_mean_spread:.6f}, predicted_class_mean_gap={predicted_class_mean_gap:.6f}, predicted_class_mean={predicted_class_mean:.6f}, other_class_mean={other_class_mean:.6f}".format(
                    feature_id=int(item["feature_id"]),
                    feature_name=feature_name,
                    feature_description=feature_description,
                    feature_type=feature_type,
                    pixel_text=pixel_text,
                    category_text=category_text,
                    variance=float(item.get("variance", 0.0)),
                    information_gain=float(item.get("information_gain", 0.0)),
                    one_feature_train_accuracy=float(item.get("one_feature_train_accuracy", 0.0)),
                    class_mean_spread=float(item.get("class_mean_spread", 0.0)),
                    predicted_class_mean_gap=float(item.get("predicted_class_mean_gap", 0.0)),
                    predicted_class_mean=float(item.get("predicted_class_mean", 0.0)),
                    other_class_mean=float(item.get("other_class_mean", 0.0)),
                )
            )
    candidate_summary_text = "\n".join(candidate_summary_lines) if candidate_summary_lines else "none"
    dataset_name = str(prompt_metadata.get("dataset", ""))
    task_intro = str(prompt_metadata.get("task_intro", "")).strip()
    target_candidates = [str(item) for item in prompt_metadata.get("target_candidates", []) if str(item).strip()]
    guidance_lines = [
        "Prefer features that are likely to create a useful one-feature split for this leaf.",
    ]
    if is_regression:
        guidance_lines = [
            "This is a regression leaf expansion task with a continuous target.",
            "Prefer features that are likely to create a threshold split with lower local prediction error.",
            "Use variance_reduction, one_feature_train_rmse, and target_mean_spread as the primary quality signals.",
            "Prefer stable threshold-like splits that separate lower-value and higher-value target regions.",
            "Do not talk about classes; focus on continuous target ranges and local error reduction.",
        ]
    if is_tabular_context:
        guidance_lines.extend(
            [
                "This is tabular business data rather than image pixels.",
                "Use feature_name and feature_type as semantic signals instead of relying on feature id order.",
                "For numeric features, prefer threshold-like splits with stable local gain.",
                "For categorical features, prefer fields whose categories are likely to isolate different target regimes or contact histories.",
                "Do not infer spatial adjacency between feature ids.",
                "Treat ordinal-encoded categorical ids as labels, not as meaningful magnitudes.",
            ]
        )
    bank_like_context = dataset_name.lower() == "bank" or "term deposit" in task_intro.lower()
    if bank_like_context:
        guidance_lines.extend(
            [
                "This bank task is about predicting term deposit subscription outcome for contacted clients.",
                "Use the predicted outcome and alternative outcomes explicitly: prefer features that either confirm the current majority outcome with a clean threshold or expose a subgroup likely to switch to the alternative outcome.",
                "Do a counterfactual check before final ranking: if two candidates both look plausible, prefer the one with stronger one_feature_train_accuracy and more stable separation rather than the one that only has larger raw magnitude differences.",
                "For example, compare age vs duration style trade-offs explicitly instead of always preferring recent-contact features.",
                "Prioritize customer-contact features such as call duration, campaign history, previous outcome, contact channel, and month when they align with the leaf statistics.",
                "Also use customer profile and financial context such as balance, housing, loan, job, education, and age when they help isolate likely subscribers or non-subscribers.",
            ]
        )
    predicted_outcome = ""
    alternative_outcomes: list[str] = []
    if target_candidates and not is_regression:
        predicted_index = int(getattr(context, "predicted_class", 0))
        if predicted_index < len(target_candidates):
            predicted_outcome = target_candidates[predicted_index]
            alternative_outcomes = [item for idx, item in enumerate(target_candidates) if idx != predicted_index]
    metadata_lines = []
    if dataset_name:
        metadata_lines.append(f"dataset={dataset_name}")
    if task_type:
        metadata_lines.append(f"task_type={task_type}")
    if task_intro:
        metadata_lines.append(f"task_intro={task_intro}")
    if target_candidates and not is_regression:
        metadata_lines.append("target_outcomes=[{}]".format(", ".join(target_candidates)))
    if predicted_outcome:
        metadata_lines.append(f"predicted_outcome={predicted_outcome}")
    if alternative_outcomes:
        metadata_lines.append("alternative_outcomes=[{}]".format(", ".join(alternative_outcomes)))
    if is_regression and getattr(context, "predicted_value", None) is not None:
        metadata_lines.append(f"predicted_value={float(context.predicted_value)}")
    if is_regression and target_summary_text:
        metadata_lines.append(f"target_summary={target_summary_text}")
    prompt_sections = [
        "You are ranking candidate features for a local leaf expansion task.\n",
        ("\n".join(metadata_lines) + "\n") if metadata_lines else "",
        f"leaf_id={context.leaf_id}\n",
        f"sample_count={context.sample_count}\n",
        f"impurity={context.impurity:.6f}\n",
    ]
    if not is_regression:
        prompt_sections.append(f"predicted_class={context.predicted_class}\n")
        if distribution_text:
            prompt_sections.append(f"class_distribution={distribution_text}\n")
    prompt_sections.extend(
        [
            f"candidate_features=[{candidate_text}]\n",
            f"candidate_feature_summaries=\n{candidate_summary_text}\n",
            "\n".join(guidance_lines) + "\n",
            "Return ONLY a JSON object with key ranked_features containing a ranked list of feature ids.\n",
            "Do not add markdown fences, explanation, or any extra keys.",
        ]
    )
    return "".join(prompt_sections)


def rank_features_with_mock_llm(context) -> list[int]:
    candidate_ids = list(context.candidate_feature_ids)
    if not candidate_ids:
        return []
    pivot = context.leaf_id % len(candidate_ids)
    rotated = candidate_ids[pivot:] + candidate_ids[:pivot]
    return [int(feature_id) for feature_id in rotated]


def parse_ranked_features_response(response_text: str) -> list[int]:
    stripped_text = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped_text, flags=re.DOTALL)
    if fenced_match:
        stripped_text = fenced_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", stripped_text, flags=re.DOTALL)
        if object_match:
            stripped_text = object_match.group(0)

    payload = json.loads(stripped_text)
    ranked_features = payload.get("ranked_features", [])
    return [int(feature_id) for feature_id in ranked_features]


def validate_ranked_features_response(ranked_features: list[int], candidate_ids: list[int]) -> None:
    candidate_pool = [int(feature_id) for feature_id in candidate_ids]
    invalid_ids = [int(feature_id) for feature_id in ranked_features if int(feature_id) not in candidate_pool]
    if invalid_ids:
        raise ValueError(f"ranked_features contains ids outside candidate_features: {invalid_ids}")

    normalized = [int(feature_id) for feature_id in ranked_features]
    if len(normalized) != len(set(normalized)):
        raise ValueError("ranked_features contains duplicate feature ids")


def _normalize_ranked_features(ranked_features: list[int], candidate_ids: list[int]) -> list[int]:
    remaining = [int(feature_id) for feature_id in candidate_ids]
    normalized: list[int] = []
    for feature_id in ranked_features:
        feature_id = int(feature_id)
        if feature_id in remaining:
            normalized.append(feature_id)
            remaining.remove(feature_id)
    normalized.extend(remaining)
    return normalized


def query_ranked_features_with_llm(
    context,
    *,
    model: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    request_interval: float = 0.0,
) -> tuple[list[int], str, str]:
    openai_settings = get_openai_settings(explicit_model=model)
    api_key = openai_settings["api_key"]
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required for relation_source=online_llm. You can set it via environment variables or DeLTa-main/local_openai_config.json"
        )

    base_url = openai_settings["base_url"]
    resolved_model = openai_settings["model"]
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = build_llm_leaf_prompt(context)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You rank features for a decision-tree leaf expansion task and must return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content or "{}"
            ranked_features = parse_ranked_features_response(response_text)
            validate_ranked_features_response(ranked_features, list(context.candidate_feature_ids))
            normalized_features = _normalize_ranked_features(ranked_features, list(context.candidate_feature_ids))
            if request_interval > 0:
                time.sleep(request_interval)
            return normalized_features, prompt, response_text
        except (APIConnectionError, APIError, BadRequestError, RateLimitError, json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(retry_delay)

    raise RuntimeError(f"Failed to query online LLM for leaf {context.leaf_id}: {last_error}")


def load_ranked_features_from_json(file_path: str | Path, selected_leaf_ids: list[int]) -> dict[int, list[int]]:
    content = json.loads(Path(file_path).read_text(encoding="utf-8"))
    ranked_features_by_leaf: dict[int, list[int]] = {}
    for leaf_id in selected_leaf_ids:
        key = str(leaf_id)
        if key in content:
            ranked_features_by_leaf[int(leaf_id)] = [int(feature_id) for feature_id in content[key]]
    return ranked_features_by_leaf