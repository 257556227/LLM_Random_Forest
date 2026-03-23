from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from model.leaf_expansion import (
    SUPPORTED_LEAF_SELECTION_STRATEGIES,
    build_llm_leaf_prompt,
    fit_leaf_expansion,
    load_ranked_features_from_json,
    query_ranked_features_with_llm,
    rank_features_with_mock_llm,
    rank_features_without_llm,
    select_candidate_leaves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST leaf expansion prototype runner")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--dataset_path", default="example_datasets")
    parser.add_argument("--base_max_depth", type=int, default=3)
    parser.add_argument("--base_max_leaf_nodes", type=int, default=20)
    parser.add_argument("--top_k_leaves", type=int, default=3)
    parser.add_argument("--min_leaf_samples", type=int, default=80)
    parser.add_argument("--top_feature_k", type=int, default=10)
    parser.add_argument("--leaf_selection_strategy", choices=list(SUPPORTED_LEAF_SELECTION_STRATEGIES), default="impurity_mass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["without_llm", "llm_guided", "both"], default="both")
    parser.add_argument("--relation_source", choices=["mock_llm", "json", "online_llm"], default="mock_llm")
    parser.add_argument("--llm_relation_file", default="")
    parser.add_argument("--llm_model", default="")
    parser.add_argument("--llm_max_retries", type=int, default=3)
    parser.add_argument("--llm_retry_delay", type=float, default=2.0)
    parser.add_argument("--llm_request_interval", type=float, default=0.0)
    parser.add_argument("--top_k_features_to_try", type=int, default=3)
    parser.add_argument("--local_feature_subset_size", type=int, default=1)
    parser.add_argument("--local_max_depth", type=int, default=1)
    parser.add_argument("--local_max_leaf_nodes", type=int, default=2)
    parser.add_argument("--save_summary", action="store_true")
    parser.add_argument("--output", default="results/mnist/leaf_expansion_without_llm_summary.json")
    parser.add_argument("--guided_output", default="results/mnist/llm_guided_leaf_expansion_summary.json")
    return parser.parse_args()


def _merge_train_val(parts: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([parts["train"], parts["val"]], axis=0)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def load_mnist_dataset(dataset: str, dataset_path: str):
    dataset_dir = Path(dataset_path) / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    numeric_parts = {}
    categorical_parts = {}
    for split in ["train", "val", "test"]:
        numeric_path = dataset_dir / f"N_{split}.npy"
        categorical_path = dataset_dir / f"C_{split}.npy"
        if numeric_path.exists():
            numeric_parts[split] = np.load(numeric_path, allow_pickle=True)
        if categorical_path.exists():
            categorical_parts[split] = np.load(categorical_path, allow_pickle=True)

    target_parts = {
        split: np.load(dataset_dir / f"y_{split}.npy", allow_pickle=True)
        for split in ["train", "val", "test"]
    }
    info = json.loads((dataset_dir / "info.json").read_text(encoding="utf-8"))

    if not numeric_parts and not categorical_parts:
        raise FileNotFoundError(f"No numeric or categorical features found under {dataset_dir}")

    if categorical_parts:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        fit_source = np.concatenate([categorical_parts["train"], categorical_parts["val"]], axis=0)
        encoder.fit(fit_source)
        encoded_categorical_parts = {
            split: encoder.transform(categorical_parts[split]).astype(float)
            for split in ["train", "val", "test"]
        }
    else:
        encoded_categorical_parts = {}

    feature_intro = info.get("feature_intro")
    if feature_intro is None:
        feature_intro = {
            "num": info.get("num_feature_intro", {}),
            "cat": info.get("cat_feature_intro", {}),
        }
    num_feature_names = list(feature_intro.get("num", {}).keys())
    cat_feature_names = list(feature_intro.get("cat", {}).keys())
    num_feature_descriptions = [str(value) for value in feature_intro.get("num", {}).values()]
    cat_feature_descriptions = [str(value) for value in feature_intro.get("cat", {}).values()]

    feature_parts = {}
    for split in ["train", "val", "test"]:
        matrices = []
        if split in numeric_parts:
            matrices.append(np.asarray(numeric_parts[split], dtype=float))
        if split in encoded_categorical_parts:
            matrices.append(np.asarray(encoded_categorical_parts[split], dtype=float))
        feature_parts[split] = np.concatenate(matrices, axis=1) if len(matrices) > 1 else matrices[0]

    if not num_feature_names and "train" in numeric_parts:
        num_feature_names = [f"num_feature_{index}" for index in range(numeric_parts["train"].shape[1])]
        num_feature_descriptions = list(num_feature_names)
    if not cat_feature_names and "train" in encoded_categorical_parts:
        cat_feature_names = [f"cat_feature_{index}" for index in range(encoded_categorical_parts["train"].shape[1])]
        cat_feature_descriptions = list(cat_feature_names)

    info["leaf_expansion_feature_names"] = num_feature_names + cat_feature_names
    info["leaf_expansion_feature_descriptions"] = num_feature_descriptions + cat_feature_descriptions
    info["leaf_expansion_feature_types"] = (["numeric"] * len(num_feature_names)) + (["categorical"] * len(cat_feature_names))
    feature_value_hints = [None for _ in range(len(num_feature_names))]
    if categorical_parts:
        for categories in encoder.categories_:
            feature_value_hints.append({"category_values": [str(value) for value in categories.tolist()]})
    info["leaf_expansion_feature_value_hints"] = feature_value_hints
    target_intro = info.get("target_intro", {})
    target_candidates = _dedupe_preserve_order(list(target_intro.keys()) + list(target_intro.values()))
    info["leaf_expansion_prompt_metadata"] = {
        "dataset": dataset,
        "task_type": info.get("task_type", ""),
        "task_intro": str(info.get("task_intro", "")),
        "target_candidates": target_candidates,
    }
    return feature_parts, target_parts, info


def build_ranked_features_without_llm(selected_contexts, X_train: np.ndarray, leaf_ids: np.ndarray) -> dict[int, list[int]]:
    ranked_features_by_leaf = {}
    for context in selected_contexts:
        indices = np.where(leaf_ids == context.leaf_id)[0]
        leaf_X = X_train[indices]
        ranked_features_by_leaf[context.leaf_id] = rank_features_without_llm(
            leaf_X,
            top_feature_ids=context.candidate_feature_ids,
        )
    return ranked_features_by_leaf


def build_ranked_features_with_llm(
    selected_contexts,
    relation_source: str,
    llm_relation_file: str,
    llm_model: str,
    llm_max_retries: int,
    llm_retry_delay: float,
    llm_request_interval: float,
) -> tuple[dict[int, list[int]], dict[int, str], dict[int, str], str | None]:
    selected_leaf_ids = [int(context.leaf_id) for context in selected_contexts]
    leaf_prompts: dict[int, str] = {}
    llm_raw_responses: dict[int, str] = {}
    resolved_llm_model: str | None = None

    if relation_source == "json":
        leaf_prompts = {int(context.leaf_id): build_llm_leaf_prompt(context) for context in selected_contexts}
        if not llm_relation_file:
            raise ValueError("--llm_relation_file is required when relation_source=json")
        ranked_features_by_leaf = load_ranked_features_from_json(llm_relation_file, selected_leaf_ids)
    elif relation_source == "online_llm":
        ranked_features_by_leaf = {}
        for context in selected_contexts:
            ranked_features, prompt, raw_response = query_ranked_features_with_llm(
                context,
                model=llm_model or None,
                max_retries=llm_max_retries,
                retry_delay=llm_retry_delay,
                request_interval=llm_request_interval,
            )
            leaf_id = int(context.leaf_id)
            ranked_features_by_leaf[leaf_id] = ranked_features
            leaf_prompts[leaf_id] = prompt
            llm_raw_responses[leaf_id] = raw_response
        resolved_llm_model = llm_model or None
    else:
        leaf_prompts = {int(context.leaf_id): build_llm_leaf_prompt(context) for context in selected_contexts}
        ranked_features_by_leaf = {
            int(context.leaf_id): rank_features_with_mock_llm(context)
            for context in selected_contexts
        }
    return ranked_features_by_leaf, leaf_prompts, llm_raw_responses, resolved_llm_model


def run_prototype(base_tree, X_train, y_train, X_test, y_test, selected_contexts, ranked_features_by_leaf, prototype_name, top_k_features_to_try=3, local_feature_subset_size=1, local_max_depth=1, local_max_leaf_nodes=2, relation_source=None, leaf_prompts=None, llm_raw_responses=None, llm_model=None, leaf_selection_strategy="impurity_mass", task_type="classification"):
    expanded_model = fit_leaf_expansion(
        base_tree=base_tree,
        X_train=X_train,
        y_train=y_train,
        selected_leaf_contexts=selected_contexts,
        ranked_features_by_leaf=ranked_features_by_leaf,
        random_state=42,
        top_k_features_to_try=top_k_features_to_try,
        local_feature_subset_size=local_feature_subset_size,
        local_max_depth=local_max_depth,
        local_max_leaf_nodes=local_max_leaf_nodes,
        task_type=task_type,
    )
    baseline_predictions = base_tree.predict(X_test)
    expanded_predictions = expanded_model.predict(X_test)
    summary = {
        "prototype": prototype_name,
        "selected_leaf_ids": [int(context.leaf_id) for context in selected_contexts],
        "expanded_leaf_count": int(len(expanded_model.leaf_expansions)),
        "ranked_features_by_leaf": {
            str(leaf_id): feature_ids for leaf_id, feature_ids in ranked_features_by_leaf.items()
        },
        "top_k_features_to_try": int(top_k_features_to_try),
        "local_feature_subset_size": int(local_feature_subset_size),
        "local_max_depth": int(local_max_depth),
        "local_max_leaf_nodes": int(local_max_leaf_nodes),
        "leaf_selection_strategy": str(leaf_selection_strategy),
        "selected_expansion_features_by_leaf": {
            str(leaf_id): (
                [int(feature_id) for feature_id in expansion.feature_ids]
                if len(expansion.feature_ids) > 1
                else int(expansion.feature_id)
            )
            for leaf_id, expansion in expanded_model.leaf_expansions.items()
        },
    }
    if task_type == "regression":
        summary["metric_name"] = "rmse"
        summary["baseline_rmse"] = float(mean_squared_error(y_test, baseline_predictions) ** 0.5)
        summary["expanded_rmse"] = float(mean_squared_error(y_test, expanded_predictions) ** 0.5)
        summary["baseline_r2"] = float(r2_score(y_test, baseline_predictions))
        summary["expanded_r2"] = float(r2_score(y_test, expanded_predictions))
    else:
        summary["metric_name"] = "accuracy"
        summary["baseline_accuracy"] = float(accuracy_score(y_test, baseline_predictions))
        summary["expanded_accuracy"] = float(accuracy_score(y_test, expanded_predictions))
    if relation_source is not None:
        summary["relation_source"] = relation_source
    if leaf_prompts is not None:
        summary["leaf_prompts"] = {str(leaf_id): prompt for leaf_id, prompt in leaf_prompts.items()}
    if llm_raw_responses:
        summary["llm_raw_responses"] = {str(leaf_id): response for leaf_id, response in llm_raw_responses.items()}
    if llm_model:
        summary["llm_model"] = llm_model
    return summary


def main():
    args = parse_args()
    numeric_parts, target_parts, info = load_mnist_dataset(args.dataset, args.dataset_path)
    X_train = _merge_train_val(numeric_parts)
    task_type = str(info.get("task_type", "classification"))
    y_train = _merge_train_val(target_parts).astype(float if task_type == "regression" else int)
    X_test = numeric_parts["test"]
    y_test = target_parts["test"].astype(float if task_type == "regression" else int)

    base_tree = (
        DecisionTreeRegressor(
            max_depth=args.base_max_depth,
            max_leaf_nodes=args.base_max_leaf_nodes,
            random_state=args.seed,
        )
        if task_type == "regression"
        else DecisionTreeClassifier(
            max_depth=args.base_max_depth,
            max_leaf_nodes=args.base_max_leaf_nodes,
            random_state=args.seed,
        )
    )
    base_tree.fit(X_train, y_train)

    selected_contexts = select_candidate_leaves(
        base_tree,
        X_train,
        y_train,
        top_k_leaves=args.top_k_leaves,
        min_leaf_samples=args.min_leaf_samples,
        top_feature_k=args.top_feature_k,
        leaf_selection_strategy=args.leaf_selection_strategy,
        feature_names=info.get("leaf_expansion_feature_names"),
        feature_descriptions=info.get("leaf_expansion_feature_descriptions"),
        feature_types=info.get("leaf_expansion_feature_types"),
        feature_value_hints=info.get("leaf_expansion_feature_value_hints"),
        prompt_metadata=info.get("leaf_expansion_prompt_metadata"),
        task_type=task_type,
    )
    leaf_ids = base_tree.apply(X_train)
    if args.mode in {"without_llm", "both"}:
        ranked_features_by_leaf = build_ranked_features_without_llm(selected_contexts, X_train, leaf_ids)
        summary = run_prototype(
            base_tree,
            X_train,
            y_train,
            X_test,
            y_test,
            selected_contexts,
            ranked_features_by_leaf,
            prototype_name="Leaf Expansion without LLM",
            top_k_features_to_try=args.top_k_features_to_try,
            local_feature_subset_size=args.local_feature_subset_size,
            local_max_depth=args.local_max_depth,
            local_max_leaf_nodes=args.local_max_leaf_nodes,
            leaf_selection_strategy=args.leaf_selection_strategy,
            task_type=task_type,
        )
        summary["dataset"] = args.dataset
        summary["task_type"] = info["task_type"]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.save_summary:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode in {"llm_guided", "both"}:
        ranked_features_by_leaf, leaf_prompts, llm_raw_responses, resolved_llm_model = build_ranked_features_with_llm(
            selected_contexts,
            relation_source=args.relation_source,
            llm_relation_file=args.llm_relation_file,
            llm_model=args.llm_model,
            llm_max_retries=args.llm_max_retries,
            llm_retry_delay=args.llm_retry_delay,
            llm_request_interval=args.llm_request_interval,
        )
        guided_summary = run_prototype(
            base_tree,
            X_train,
            y_train,
            X_test,
            y_test,
            selected_contexts,
            ranked_features_by_leaf,
            prototype_name="LLM-Guided Leaf Expansion",
            top_k_features_to_try=args.top_k_features_to_try,
            local_feature_subset_size=args.local_feature_subset_size,
            local_max_depth=args.local_max_depth,
            local_max_leaf_nodes=args.local_max_leaf_nodes,
            relation_source=args.relation_source,
            leaf_prompts=leaf_prompts,
            llm_raw_responses=llm_raw_responses,
            llm_model=resolved_llm_model,
            leaf_selection_strategy=args.leaf_selection_strategy,
            task_type=task_type,
        )
        guided_summary["dataset"] = args.dataset
        guided_summary["task_type"] = info["task_type"]
        print(json.dumps(guided_summary, ensure_ascii=False, indent=2))
        if args.save_summary:
            output_path = Path(args.guided_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(guided_summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()