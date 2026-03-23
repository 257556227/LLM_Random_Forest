from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _aligned_proba(model: DecisionTreeClassifier, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw_proba = model.predict_proba(X)
    if isinstance(raw_proba, list):
        raw_proba = raw_proba[0]
    aligned = np.zeros((X.shape[0], n_classes), dtype=float)
    for position, class_id in enumerate(model.classes_):
        aligned[:, int(class_id)] = raw_proba[:, position]
    return aligned


@dataclass
class ExpandedLeafModel:
    base_tree: object
    leaf_expansions: dict[int, object]
    n_classes: int | None = None
    task_type: str = "classification"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "regression":
            raise AttributeError("predict_proba is only available for classification tasks")
        base_leaf_ids = self.base_tree.apply(X)
        base_proba = self.base_tree.predict_proba(X)
        if isinstance(base_proba, list):
            base_proba = base_proba[0]
        predictions = np.array(base_proba, copy=True)

        for leaf_id, expansion_model in self.leaf_expansions.items():
            indices = np.where(base_leaf_ids == leaf_id)[0]
            if len(indices) == 0:
                continue
            predictions[indices] = _aligned_proba(expansion_model, X[indices], self.n_classes)
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "regression":
            base_leaf_ids = self.base_tree.apply(X)
            predictions = np.array(self.base_tree.predict(X), copy=True, dtype=float)
            for leaf_id, expansion_model in self.leaf_expansions.items():
                indices = np.where(base_leaf_ids == leaf_id)[0]
                if len(indices) == 0:
                    continue
                predictions[indices] = expansion_model.predict(X[indices])
            return predictions
        return np.argmax(self.predict_proba(X), axis=1)


def fit_leaf_expansion(
    base_tree: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_leaf_contexts,
    ranked_features_by_leaf: dict[int, list[int]],
    random_state: int = 42,
    top_k_features_to_try: int = 3,
    local_feature_subset_size: int = 1,
    local_max_depth: int = 1,
    local_max_leaf_nodes: int = 2,
    task_type: str = "classification",
) -> ExpandedLeafModel:
    leaf_ids = base_tree.apply(X_train)
    leaf_expansions: dict[int, object] = {}
    n_classes = None if task_type == "regression" else len(np.unique(y_train))

    for context in selected_leaf_contexts:
        indices = np.where(leaf_ids == context.leaf_id)[0]
        if len(indices) < 2:
            continue

        leaf_X = X_train[indices]
        leaf_y = y_train[indices]
        if np.unique(leaf_y).shape[0] < 2:
            continue

        ranked_features = ranked_features_by_leaf.get(context.leaf_id, context.candidate_feature_ids)
        if len(ranked_features) == 0:
            continue

        feature_candidates = [int(feature_id) for feature_id in ranked_features[: max(1, top_k_features_to_try)]]
        subset_limit = max(1, min(int(local_feature_subset_size), len(feature_candidates)))
        candidate_feature_sets: list[tuple[int, ...]] = []
        for subset_size in range(1, subset_limit + 1):
            candidate_feature_sets.extend(combinations(feature_candidates, subset_size))

        selected_feature_ids = list(candidate_feature_sets[0])
        best_feature_score = -1.0 if task_type != "regression" else float("inf")
        for feature_id_set in candidate_feature_sets:
            local_X = leaf_X[:, list(feature_id_set)]
            candidate_tree = _build_local_tree(
                task_type=task_type,
                local_max_depth=local_max_depth,
                local_max_leaf_nodes=local_max_leaf_nodes,
                random_state=random_state,
            )
            candidate_tree.fit(local_X, leaf_y)
            if task_type == "regression":
                candidate_score = float(np.sqrt(np.mean((candidate_tree.predict(local_X) - leaf_y) ** 2)))
                if candidate_score < best_feature_score:
                    best_feature_score = candidate_score
                    selected_feature_ids = [int(feature_id) for feature_id in feature_id_set]
            else:
                candidate_score = float(np.mean(candidate_tree.predict(local_X) == leaf_y))
                if candidate_score > best_feature_score:
                    best_feature_score = candidate_score
                    selected_feature_ids = [int(feature_id) for feature_id in feature_id_set]

        local_X = leaf_X[:, selected_feature_ids]
        wrapped_tree = _build_local_tree(
            task_type=task_type,
            local_max_depth=local_max_depth,
            local_max_leaf_nodes=local_max_leaf_nodes,
            random_state=random_state,
        )
        wrapped_tree.fit(local_X, leaf_y)
        leaf_expansions[int(context.leaf_id)] = _FeatureSubsetExpansionModel(
            feature_ids=selected_feature_ids,
            model=wrapped_tree,
        )

    return ExpandedLeafModel(base_tree=base_tree, leaf_expansions=leaf_expansions, n_classes=n_classes, task_type=task_type)


def _build_local_tree(task_type: str, local_max_depth: int, local_max_leaf_nodes: int, random_state: int):
    if task_type == "regression":
        return DecisionTreeRegressor(
            max_depth=max(1, int(local_max_depth)),
            max_leaf_nodes=max(2, int(local_max_leaf_nodes)),
            random_state=random_state,
        )
    return DecisionTreeClassifier(
        max_depth=max(1, int(local_max_depth)),
        max_leaf_nodes=max(2, int(local_max_leaf_nodes)),
        random_state=random_state,
    )


@dataclass
class _FeatureSubsetExpansionModel:
    feature_ids: list[int]
    model: object

    @property
    def feature_id(self):
        return self.feature_ids[0]

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X[:, self.feature_ids])

    def predict(self, X: np.ndarray):
        return self.model.predict(X[:, self.feature_ids])