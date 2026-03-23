from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SUPPORTED_LEAF_SELECTION_STRATEGIES = (
    "impurity_mass",
    "impurity",
    "sample_count",
)


@dataclass
class LeafContext:
    leaf_id: int
    sample_count: int
    impurity: float
    candidate_feature_ids: list[int] = field(default_factory=list)
    predicted_class: int | None = None
    class_distribution: dict[int, int] = field(default_factory=dict)
    predicted_value: float | None = None
    target_summary: dict[str, float] = field(default_factory=dict)
    candidate_feature_summaries: list[dict[str, float | int]] = field(default_factory=list)
    prompt_metadata: dict[str, object] = field(default_factory=dict)


def _leaf_class_distribution(labels: np.ndarray) -> dict[int, int]:
    counter = Counter(labels.tolist())
    return {int(key): int(value) for key, value in counter.items()}


def _looks_like_image_features(feature_names: list[str] | None, n_features: int) -> bool:
    side_length = int(np.sqrt(n_features))
    if side_length * side_length != n_features:
        return False

    if not feature_names:
        return True

    normalized_names = [str(name).strip().lower() for name in feature_names]
    image_like_prefixes = ("pixel", "px")
    return all(name.isdigit() or name.startswith(image_like_prefixes) for name in normalized_names)


def _infer_pixel_coordinates(
    feature_id: int,
    n_features: int,
    feature_names: list[str] | None = None,
) -> tuple[int | None, int | None]:
    if not _looks_like_image_features(feature_names, n_features):
        return None, None

    side_length = int(np.sqrt(n_features))
    if side_length * side_length != n_features:
        return None, None
    return int(feature_id // side_length), int(feature_id % side_length)


def _single_feature_information_gain(feature_values: np.ndarray, labels: np.ndarray) -> float:
    if feature_values.ndim != 2 or feature_values.shape[1] != 1:
        raise ValueError("feature_values must be shaped as (n_samples, 1)")
    if labels.shape[0] < 2 or np.unique(labels).shape[0] < 2:
        return 0.0

    stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=42)
    stump.fit(feature_values, labels)
    tree_state = stump.tree_
    if tree_state.node_count <= 1:
        return 0.0

    root_impurity = float(tree_state.impurity[0])
    root_samples = float(tree_state.n_node_samples[0])
    weighted_children_impurity = 0.0
    for child_id in (1, 2):
        if child_id >= tree_state.node_count:
            continue
        child_weight = float(tree_state.n_node_samples[child_id]) / root_samples
        weighted_children_impurity += child_weight * float(tree_state.impurity[child_id])
    return max(0.0, root_impurity - weighted_children_impurity)


def _single_feature_train_accuracy(feature_values: np.ndarray, labels: np.ndarray) -> float:
    if feature_values.ndim != 2 or feature_values.shape[1] != 1:
        raise ValueError("feature_values must be shaped as (n_samples, 1)")
    if labels.shape[0] < 2:
        return 0.0

    stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=42)
    stump.fit(feature_values, labels)
    predictions = stump.predict(feature_values)
    return float(np.mean(predictions == labels))


def _single_feature_variance_reduction(feature_values: np.ndarray, labels: np.ndarray) -> float:
    if feature_values.ndim != 2 or feature_values.shape[1] != 1:
        raise ValueError("feature_values must be shaped as (n_samples, 1)")
    if labels.shape[0] < 2:
        return 0.0

    stump = DecisionTreeRegressor(max_depth=1, max_leaf_nodes=2, random_state=42)
    stump.fit(feature_values, labels)
    tree_state = stump.tree_
    if tree_state.node_count <= 1:
        return 0.0

    root_impurity = float(tree_state.impurity[0])
    root_samples = float(tree_state.n_node_samples[0])
    weighted_children_impurity = 0.0
    for child_id in (1, 2):
        if child_id >= tree_state.node_count:
            continue
        child_weight = float(tree_state.n_node_samples[child_id]) / root_samples
        weighted_children_impurity += child_weight * float(tree_state.impurity[child_id])
    return max(0.0, root_impurity - weighted_children_impurity)


def _single_feature_train_rmse(feature_values: np.ndarray, labels: np.ndarray) -> float:
    if feature_values.ndim != 2 or feature_values.shape[1] != 1:
        raise ValueError("feature_values must be shaped as (n_samples, 1)")
    if labels.shape[0] < 2:
        return 0.0

    stump = DecisionTreeRegressor(max_depth=1, max_leaf_nodes=2, random_state=42)
    stump.fit(feature_values, labels)
    predictions = stump.predict(feature_values)
    return float(np.sqrt(np.mean((predictions - labels) ** 2)))


def _summarize_candidate_feature(
    leaf_X: np.ndarray,
    leaf_y: np.ndarray,
    feature_id: int,
    predicted_class: int | None,
    feature_names: list[str] | None = None,
    feature_descriptions: list[str] | None = None,
    feature_types: list[str] | None = None,
    feature_value_hints: list[dict[str, object] | None] | None = None,
    task_type: str = "classification",
    predicted_value: float | None = None,
) -> dict[str, float | int]:
    values = leaf_X[:, int(feature_id)]
    variance = float(np.var(values))
    predicted_mean = float(np.mean(values))
    other_means: list[float] = []
    class_mean_spread = 0.0
    predicted_class_mean_gap = 0.0
    if task_type != "regression":
        unique_classes = sorted(int(class_id) for class_id in np.unique(leaf_y))
        class_means = {
            class_id: float(np.mean(values[leaf_y == class_id]))
            for class_id in unique_classes
        }
        mean_values = list(class_means.values())
        class_mean_spread = float(max(mean_values) - min(mean_values)) if mean_values else 0.0

        predicted_mean = class_means.get(int(predicted_class or 0), float(np.mean(values)))
        other_means = [mean for class_id, mean in class_means.items() if int(class_id) != int(predicted_class or 0)]
        if other_means:
            predicted_class_mean_gap = float(abs(predicted_mean - float(np.mean(other_means))))

    else:
        threshold = float(np.median(values))
        lower_targets = leaf_y[values <= threshold]
        upper_targets = leaf_y[values > threshold]
        lower_mean = float(np.mean(lower_targets)) if lower_targets.size else float(np.mean(leaf_y))
        upper_mean = float(np.mean(upper_targets)) if upper_targets.size else float(np.mean(leaf_y))
        class_mean_spread = float(abs(upper_mean - lower_mean))
        predicted_center = float(predicted_value if predicted_value is not None else np.mean(leaf_y))
        predicted_class_mean_gap = float(abs(predicted_center - np.mean([lower_mean, upper_mean])))
        predicted_mean = lower_mean if abs(lower_mean - predicted_center) <= abs(upper_mean - predicted_center) else upper_mean
        other_means = [upper_mean] if predicted_mean == lower_mean else [lower_mean]

    pixel_row, pixel_col = _infer_pixel_coordinates(
        int(feature_id),
        leaf_X.shape[1],
        feature_names=feature_names,
    )
    feature_name = str(feature_id) if not feature_names else str(feature_names[int(feature_id)])
    feature_description = feature_name
    if feature_descriptions and int(feature_id) < len(feature_descriptions):
        feature_description = str(feature_descriptions[int(feature_id)])
    feature_type = "numeric"
    if feature_types and int(feature_id) < len(feature_types):
        feature_type = str(feature_types[int(feature_id)])
    # Only compute information_gain for classification tasks (regression uses variance_reduction)
    if task_type == "regression":
        information_gain_value = 0.0
        one_feature_train_accuracy_value = 0.0
    else:
        information_gain_value = _single_feature_information_gain(values.reshape(-1, 1), leaf_y)
        one_feature_train_accuracy_value = _single_feature_train_accuracy(values.reshape(-1, 1), leaf_y)

    summary = {
        "feature_id": int(feature_id),
        "feature_name": feature_name,
        "feature_description": feature_description,
        "feature_type": feature_type,
        "pixel_row": -1 if pixel_row is None else int(pixel_row),
        "pixel_col": -1 if pixel_col is None else int(pixel_col),
        "variance": variance,
        "information_gain": information_gain_value,
        "one_feature_train_accuracy": one_feature_train_accuracy_value,
        "variance_reduction": _single_feature_variance_reduction(values.reshape(-1, 1), leaf_y),
        "one_feature_train_rmse": _single_feature_train_rmse(values.reshape(-1, 1), leaf_y),
        "class_mean_spread": class_mean_spread,
        "target_mean_spread": class_mean_spread,
        "predicted_class_mean_gap": predicted_class_mean_gap,
        "predicted_value_gap": predicted_class_mean_gap,
        "predicted_class_mean": float(predicted_mean),
        "predicted_value_mean": float(predicted_mean),
        "other_class_mean": float(np.mean(other_means)) if other_means else float(predicted_mean),
        "other_value_mean": float(np.mean(other_means)) if other_means else float(predicted_mean),
    }
    if task_type != "regression" and feature_type == "categorical" and feature_value_hints and int(feature_id) < len(feature_value_hints):
        feature_hint = feature_value_hints[int(feature_id)] or {}
        category_values = [str(item) for item in feature_hint.get("category_values", [])]
        encoded_to_label = {index: label for index, label in enumerate(category_values)}

        encoded_ints = values.astype(int)
        unique_encoded, counts = np.unique(encoded_ints, return_counts=True)
        sorted_pairs = sorted(zip(unique_encoded.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)
        category_value_preview = [encoded_to_label.get(int(code), str(code)) for code, _ in sorted_pairs[:4]]

        predicted_mask = leaf_y == predicted_class
        predicted_encoded = values[predicted_mask].astype(int) if np.any(predicted_mask) else np.array([], dtype=int)
        if predicted_encoded.size:
            pred_codes, pred_counts = np.unique(predicted_encoded, return_counts=True)
            pred_pairs = sorted(zip(pred_codes.tolist(), pred_counts.tolist()), key=lambda item: item[1], reverse=True)
            predicted_class_top_categories = [encoded_to_label.get(int(code), str(code)) for code, _ in pred_pairs[:3]]
        else:
            predicted_class_top_categories = []

        other_mask = leaf_y != predicted_class
        other_encoded = values[other_mask].astype(int) if np.any(other_mask) else np.array([], dtype=int)
        if other_encoded.size:
            other_codes, other_counts = np.unique(other_encoded, return_counts=True)
            other_pairs = sorted(zip(other_codes.tolist(), other_counts.tolist()), key=lambda item: item[1], reverse=True)
            other_class_top_categories = [encoded_to_label.get(int(code), str(code)) for code, _ in other_pairs[:3]]
        else:
            other_class_top_categories = []

        summary["category_values_preview"] = category_value_preview
        summary["predicted_class_top_categories"] = predicted_class_top_categories
        summary["other_class_top_categories"] = other_class_top_categories
    return summary


def build_leaf_contexts(
    tree_model,
    X: np.ndarray,
    y: np.ndarray,
    top_feature_k: int = 10,
    feature_names: list[str] | None = None,
    feature_descriptions: list[str] | None = None,
    feature_types: list[str] | None = None,
    feature_value_hints: list[dict[str, object] | None] | None = None,
    prompt_metadata: dict[str, object] | None = None,
    task_type: str = "classification",
) -> list[LeafContext]:
    leaf_ids = tree_model.apply(X)
    tree_state = tree_model.tree_
    contexts: list[LeafContext] = []

    unique_leaf_ids = np.unique(leaf_ids)
    for leaf_id in unique_leaf_ids:
        indices = np.where(leaf_ids == leaf_id)[0]
        leaf_X = X[indices]
        leaf_y = y[indices]
        variances = np.var(leaf_X, axis=0)
        feature_order = np.argsort(-variances)[:top_feature_k]

        predicted_class = None
        predicted_value = None
        class_distribution: dict[int, int] = {}
        target_summary: dict[str, float] = {}
        if task_type == "regression":
            predicted_value = float(tree_model.predict(leaf_X[:1])[0]) if len(leaf_X) else 0.0
            target_summary = {
                "mean": float(np.mean(leaf_y)),
                "std": float(np.std(leaf_y)),
                "min": float(np.min(leaf_y)),
                "max": float(np.max(leaf_y)),
            }
        else:
            value = tree_state.value[leaf_id][0]
            predicted_class = int(np.argmax(value))
            class_distribution = _leaf_class_distribution(leaf_y)

        candidate_feature_ids = [int(feature_id) for feature_id in feature_order.tolist()]
        contexts.append(
            LeafContext(
                leaf_id=int(leaf_id),
                sample_count=int(len(indices)),
                impurity=float(tree_state.impurity[leaf_id]),
                predicted_class=predicted_class,
                class_distribution=class_distribution,
                predicted_value=predicted_value,
                target_summary=target_summary,
                candidate_feature_ids=candidate_feature_ids,
                candidate_feature_summaries=[
                    _summarize_candidate_feature(
                        leaf_X,
                        leaf_y,
                        feature_id,
                        predicted_class,
                        feature_names=feature_names,
                        feature_descriptions=feature_descriptions,
                        feature_types=feature_types,
                        feature_value_hints=feature_value_hints,
                        task_type=task_type,
                        predicted_value=predicted_value,
                    )
                    for feature_id in candidate_feature_ids
                ],
                prompt_metadata=dict(prompt_metadata or {}),
            )
        )
    return contexts


def select_candidate_leaves(
    tree_model,
    X: np.ndarray,
    y: np.ndarray,
    top_k_leaves: int = 3,
    min_leaf_samples: int = 40,
    top_feature_k: int = 10,
    leaf_selection_strategy: str = "impurity_mass",
    feature_names: list[str] | None = None,
    feature_descriptions: list[str] | None = None,
    feature_types: list[str] | None = None,
    feature_value_hints: list[dict[str, object] | None] | None = None,
    prompt_metadata: dict[str, object] | None = None,
    task_type: str = "classification",
) -> list[LeafContext]:
    if leaf_selection_strategy not in SUPPORTED_LEAF_SELECTION_STRATEGIES:
        raise ValueError(
            f"Unsupported leaf_selection_strategy={leaf_selection_strategy!r}. "
            f"Expected one of {SUPPORTED_LEAF_SELECTION_STRATEGIES}."
        )

    contexts = build_leaf_contexts(
        tree_model,
        X,
        y,
        top_feature_k=top_feature_k,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        feature_types=feature_types,
        feature_value_hints=feature_value_hints,
        prompt_metadata=prompt_metadata,
        task_type=task_type,
    )
    filtered_contexts = [context for context in contexts if context.sample_count >= min_leaf_samples]
    if leaf_selection_strategy == "impurity":
        ranking_key = lambda item: (item.impurity, item.sample_count)
    elif leaf_selection_strategy == "sample_count":
        ranking_key = lambda item: (item.sample_count, item.impurity)
    else:
        ranking_key = lambda item: (item.impurity * item.sample_count, item.impurity, item.sample_count)
    scored_contexts = sorted(
        filtered_contexts,
        key=ranking_key,
        reverse=True,
    )
    return scored_contexts[:top_k_leaves]