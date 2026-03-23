import unittest
import sys
from pathlib import Path
import importlib.util

import numpy as np
from sklearn.metrics import mean_squared_error
 
ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

from model.leaf_expansion.mnist_leaf_expander import fit_leaf_expansion
from model.leaf_expansion.mnist_leaf_selector import LeafContext, select_candidate_leaves
from model.leaf_expansion.mnist_relation_provider import (
    build_llm_leaf_prompt,
    validate_ranked_features_response,
)

REPEAT_TOOL_PATH = DELTA_ROOT / "tools" / "repeat_mnist_online_llm_same_metric.py"
repeat_spec = importlib.util.spec_from_file_location("repeat_mnist_online_llm_same_metric", REPEAT_TOOL_PATH)
repeat_module = importlib.util.module_from_spec(repeat_spec)
assert repeat_spec is not None and repeat_spec.loader is not None
repeat_spec.loader.exec_module(repeat_module)
build_repeatability_summary = repeat_module.build_repeatability_summary


class TestLeafExpansionImprovements(unittest.TestCase):
    def test_select_candidate_leaves_supports_sample_count_strategy(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.1],
                [0.0, 0.2],
                [1.0, 1.0],
                [1.0, 1.1],
                [2.0, 2.0],
            ]
        )
        y = np.array([0, 0, 0, 1, 1, 1])

        class StubTree:
            def __init__(self):
                self.tree_ = type(
                    "TreeState",
                    (),
                    {
                        "impurity": np.array([0.0, 0.0, 0.0, 0.4, 0.1, 0.8]),
                        "value": np.array(
                            [
                                [[0, 0]],
                                [[0, 0]],
                                [[0, 0]],
                                [[3, 0]],
                                [[0, 2]],
                                [[0, 1]],
                            ]
                        ),
                    },
                )()

            def apply(self, features):
                return np.array([3, 3, 3, 4, 4, 5])

        contexts = select_candidate_leaves(
            StubTree(),
            X,
            y,
            top_k_leaves=2,
            min_leaf_samples=1,
            top_feature_k=1,
            leaf_selection_strategy="sample_count",
        )

        self.assertEqual([context.leaf_id for context in contexts], [3, 4])

    def test_build_llm_leaf_prompt_contains_richer_feature_descriptions(self):
        context = LeafContext(
            leaf_id=8,
            sample_count=100,
            impurity=0.5,
            predicted_class=2,
            class_distribution={2: 60, 7: 40},
            candidate_feature_ids=[374, 516],
            candidate_feature_summaries=[
                {
                    "feature_id": 374,
                    "pixel_row": 13,
                    "pixel_col": 10,
                    "variance": 0.41,
                    "information_gain": 0.08,
                    "class_mean_spread": 0.32,
                    "predicted_class_mean_gap": 0.21,
                },
                {
                    "feature_id": 516,
                    "pixel_row": 18,
                    "pixel_col": 12,
                    "variance": 0.39,
                    "information_gain": 0.05,
                    "class_mean_spread": 0.18,
                    "predicted_class_mean_gap": 0.11,
                },
            ],
        )

        prompt = build_llm_leaf_prompt(context)
        self.assertIn("candidate_feature_summaries", prompt)
        self.assertIn("feature_id=374", prompt)
        self.assertIn("pixel=(13,10)", prompt)
        self.assertIn("information_gain=0.080000", prompt)
        self.assertIn("class_mean_spread=0.320000", prompt)
        self.assertIn("predicted_class_mean_gap=0.210000", prompt)

    def test_validate_ranked_features_response_rejects_out_of_candidate_ids(self):
        with self.assertRaises(ValueError):
            validate_ranked_features_response([516, 354, 484], [484, 516, 374])

    def test_validate_ranked_features_response_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            validate_ranked_features_response([374, 374, 484], [374, 484, 516])

    def test_build_llm_leaf_prompt_uses_tabular_guidance_for_bank_like_context(self):
        context = LeafContext(
            leaf_id=5,
            sample_count=30388,
            impurity=0.110522,
            predicted_class=0,
            class_distribution={0: 28604, 1: 1784},
            candidate_feature_ids=[1, 3, 7, 15],
            candidate_feature_summaries=[
                {
                    "feature_id": 1,
                    "feature_name": "balance",
                    "feature_description": "account balance",
                    "feature_type": "numeric",
                    "pixel_row": -1,
                    "pixel_col": -1,
                    "variance": 8631700.401614,
                    "information_gain": 0.000684,
                    "class_mean_spread": 579.604093,
                    "predicted_class_mean_gap": 579.604093,
                    "one_feature_train_accuracy": 0.612341,
                },
                {
                    "feature_id": 3,
                    "feature_name": "duration",
                    "feature_description": "last contact duration",
                    "feature_type": "numeric",
                    "pixel_row": -1,
                    "pixel_col": -1,
                    "variance": 13533.220427,
                    "information_gain": 0.003340,
                    "class_mean_spread": 101.127624,
                    "predicted_class_mean_gap": 101.127624,
                    "one_feature_train_accuracy": 0.604781,
                },
                {
                    "feature_id": 7,
                    "feature_name": "job",
                    "feature_description": "job type",
                    "feature_type": "categorical",
                    "category_values_preview": ["admin.", "technician", "services"],
                    "predicted_class_top_categories": ["admin.", "technician"],
                    "other_class_top_categories": ["services", "blue-collar"],
                    "pixel_row": -1,
                    "pixel_col": -1,
                    "variance": 10.926133,
                    "information_gain": 0.000421,
                    "class_mean_spread": 0.593389,
                    "predicted_class_mean_gap": 0.593389,
                    "one_feature_train_accuracy": 0.617498,
                },
                {
                    "feature_id": 15,
                    "feature_name": "poutcome",
                    "feature_description": "outcome of previous marketing campaign",
                    "feature_type": "categorical",
                    "pixel_row": -1,
                    "pixel_col": -1,
                    "variance": 0.970504,
                    "information_gain": 0.003707,
                    "class_mean_spread": 0.399213,
                    "predicted_class_mean_gap": 0.399213,
                    "one_feature_train_accuracy": 0.616414,
                },
            ],
            prompt_metadata={
                "dataset": "bank",
                "task_type": "binclass",
                "task_intro": "Predict whether a contacted bank client will subscribe a term deposit.",
                "target_candidates": [
                    "the product (bank term deposit) would be subscribed",
                    "the product (bank term deposit) would not be subscribed",
                ],
            },
        )

        prompt = build_llm_leaf_prompt(context)
        self.assertIn("tabular business data", prompt)
        self.assertIn("numeric features", prompt)
        self.assertIn("categorical features", prompt)
        self.assertIn("term deposit", prompt)
        self.assertIn("subscription outcome", prompt)
        self.assertIn("target_outcomes", prompt)
        self.assertIn("predicted_outcome", prompt)
        self.assertIn("alternative_outcomes", prompt)
        self.assertIn("category_values_preview=['admin.', 'technician', 'services']", prompt)
        self.assertIn("predicted_class_top_categories=['admin.', 'technician']", prompt)
        self.assertIn("one_feature_train_accuracy=0.612341", prompt)
        self.assertIn("Do a counterfactual check", prompt)
        self.assertIn("age vs duration", prompt)
        self.assertIn("feature_description=account balance", prompt)
        self.assertIn("feature_name=balance", prompt)
        self.assertNotIn("pixel=(0,1)", prompt)
        self.assertNotIn("spatial", prompt.lower().split("candidate_feature_summaries=")[0])

    def test_build_repeatability_summary_accepts_custom_repeat_count(self):
        original_run_once = repeat_module._run_once
        try:
            payloads = iter(
                [
                    {"status": "success", "test_accuracy": 0.5622, "llm_model": "gpt-4o"},
                    {"status": "success", "test_accuracy": 0.5622, "llm_model": "gpt-4o"},
                    {"status": "success", "test_accuracy": 0.5622, "llm_model": "gpt-4o"},
                    {"status": "success", "test_accuracy": 0.5622, "llm_model": "gpt-4o"},
                    {"status": "success", "test_accuracy": 0.5622, "llm_model": "gpt-4o"},
                ]
            )

            def fake_run_once() -> dict:
                return next(payloads)

            repeat_module._run_once = fake_run_once
            summary = build_repeatability_summary(repeats=5)
        finally:
            repeat_module._run_once = original_run_once

        self.assertEqual(summary["repeats"], 5)
        self.assertEqual(summary["success_count"], 5)
        self.assertEqual(len(summary["runs"]), 5)
        self.assertEqual(summary["wins_over_mock_count"], 5)

    def test_build_llm_leaf_prompt_uses_regression_guidance_for_house_context(self):
        context = LeafContext(
            leaf_id=3,
            sample_count=128,
            impurity=0.248,
            predicted_value=142500.0,
            target_summary={
                "mean": 142500.0,
                "std": 18500.0,
                "min": 96000.0,
                "max": 188000.0,
            },
            candidate_feature_ids=[0, 12],
            candidate_feature_summaries=[
                {
                    "feature_id": 0,
                    "feature_name": "P1",
                    "feature_description": "demographic proportion feature P1",
                    "feature_type": "numeric",
                    "variance": 0.41,
                    "variance_reduction": 0.081,
                    "one_feature_train_rmse": 13200.0,
                    "target_mean_spread": 28600.0,
                    "predicted_value_gap": 9100.0,
                    "predicted_value_mean": 147800.0,
                    "other_value_mean": 138700.0,
                },
                {
                    "feature_id": 12,
                    "feature_name": "H10p1",
                    "feature_description": "housing market feature H10p1",
                    "feature_type": "numeric",
                    "variance": 0.23,
                    "variance_reduction": 0.056,
                    "one_feature_train_rmse": 13950.0,
                    "target_mean_spread": 24100.0,
                    "predicted_value_gap": 6200.0,
                    "predicted_value_mean": 145100.0,
                    "other_value_mean": 138900.0,
                },
            ],
            prompt_metadata={
                "dataset": "house_16H_reg",
                "task_type": "regression",
                "task_intro": "Predict the median house price for a region from 16 continuous demographic and housing market features.",
            },
        )

        prompt = build_llm_leaf_prompt(context)
        self.assertIn("task_type=regression", prompt)
        self.assertIn("predicted_value=142500.0", prompt)
        self.assertIn("target_summary=mean: 142500.000000", prompt)
        self.assertIn("variance_reduction=0.081000", prompt)
        self.assertIn("one_feature_train_rmse=13200.000000", prompt)
        self.assertIn("continuous target", prompt)
        self.assertIn("lower local prediction error", prompt)

    def test_fit_leaf_expansion_supports_regression_local_tree(self):
        X_train = np.array(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        )
        y_train = np.array([100.0, 120.0, 180.0, 200.0])

        class StubRegressor:
            def apply(self, X):
                return np.full(X.shape[0], 8, dtype=int)

            def predict(self, X):
                return np.full(X.shape[0], 150.0, dtype=float)

        context = LeafContext(
            leaf_id=8,
            sample_count=4,
            impurity=400.0,
            predicted_value=150.0,
            target_summary={"mean": 150.0, "std": 42.4264, "min": 100.0, "max": 200.0},
            candidate_feature_ids=[0],
            candidate_feature_summaries=[],
            prompt_metadata={"dataset": "house_16H_reg", "task_type": "regression"},
        )

        model = fit_leaf_expansion(
            base_tree=StubRegressor(),
            X_train=X_train,
            y_train=y_train,
            selected_leaf_contexts=[context],
            ranked_features_by_leaf={8: [0]},
            random_state=42,
            top_k_features_to_try=1,
            local_feature_subset_size=1,
            local_max_depth=2,
            local_max_leaf_nodes=4,
            task_type="regression",
        )

        baseline_rmse = mean_squared_error(y_train, StubRegressor().predict(X_train)) ** 0.5
        expanded_rmse = mean_squared_error(y_train, model.predict(X_train)) ** 0.5

        self.assertLess(expanded_rmse, baseline_rmse)
        self.assertEqual(model.leaf_expansions[8].feature_ids, [0])

    def test_fit_leaf_expansion_can_choose_best_feature_within_top_k(self):
        X_train = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )
        y_train = np.array([0, 0, 1, 1, 0, 1])

        class StubTree:
            def apply(self, X):
                return np.full(X.shape[0], 8, dtype=int)

            def predict_proba(self, X):
                return np.full((X.shape[0], 3), 1 / 3, dtype=float)

        base_tree = StubTree()
        context = LeafContext(
            leaf_id=8,
            sample_count=6,
            impurity=0.5,
            predicted_class=0,
            class_distribution={0: 3, 1: 3},
            candidate_feature_ids=[0, 1],
            candidate_feature_summaries=[],
        )

        ranked_features = {8: [0, 1]}
        improved_model = fit_leaf_expansion(
            base_tree=base_tree,
            X_train=X_train,
            y_train=y_train,
            selected_leaf_contexts=[context],
            ranked_features_by_leaf=ranked_features,
            random_state=42,
            top_k_features_to_try=2,
        )

        self.assertIn(context.leaf_id, improved_model.leaf_expansions)
        self.assertEqual(improved_model.leaf_expansions[context.leaf_id].feature_id, 1)

    def test_fit_leaf_expansion_supports_deeper_local_split(self):
        X_train = np.array(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
            ]
        )
        y_train = np.array([0, 0, 1, 1, 2, 2])

        class StubTree:
            def apply(self, X):
                return np.full(X.shape[0], 8, dtype=int)

            def predict_proba(self, X):
                return np.full((X.shape[0], 3), 1 / 3, dtype=float)

        context = LeafContext(
            leaf_id=8,
            sample_count=6,
            impurity=0.66,
            predicted_class=0,
            class_distribution={0: 2, 1: 2, 2: 2},
            candidate_feature_ids=[0],
            candidate_feature_summaries=[],
        )

        depth1_model = fit_leaf_expansion(
            base_tree=StubTree(),
            X_train=X_train,
            y_train=y_train,
            selected_leaf_contexts=[context],
            ranked_features_by_leaf={8: [0]},
            random_state=42,
            top_k_features_to_try=1,
            local_max_depth=1,
            local_max_leaf_nodes=2,
        )
        depth2_model = fit_leaf_expansion(
            base_tree=StubTree(),
            X_train=X_train,
            y_train=y_train,
            selected_leaf_contexts=[context],
            ranked_features_by_leaf={8: [0]},
            random_state=42,
            top_k_features_to_try=1,
            local_max_depth=2,
            local_max_leaf_nodes=3,
        )

        depth1_accuracy = np.mean(depth1_model.predict(X_train) == y_train)
        depth2_accuracy = np.mean(depth2_model.predict(X_train) == y_train)

        self.assertLess(depth1_accuracy, 1.0)
        self.assertEqual(depth2_accuracy, 1.0)

    def test_fit_leaf_expansion_can_choose_multi_feature_local_structure(self):
        X_train = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        y_train = np.array([0, 1, 1, 0])

        class StubTree:
            def apply(self, X):
                return np.full(X.shape[0], 8, dtype=int)

            def predict_proba(self, X):
                return np.full((X.shape[0], 2), 0.5, dtype=float)

        context = LeafContext(
            leaf_id=8,
            sample_count=4,
            impurity=0.5,
            predicted_class=0,
            class_distribution={0: 2, 1: 2},
            candidate_feature_ids=[0, 1],
            candidate_feature_summaries=[],
        )

        single_feature_model = fit_leaf_expansion(
            base_tree=StubTree(),
            X_train=X_train,
            y_train=y_train,
            selected_leaf_contexts=[context],
            ranked_features_by_leaf={8: [0, 1]},
            random_state=42,
            top_k_features_to_try=2,
            local_feature_subset_size=1,
            local_max_depth=2,
            local_max_leaf_nodes=4,
        )
        multi_feature_model = fit_leaf_expansion(
            base_tree=StubTree(),
            X_train=X_train,
            y_train=y_train,
            selected_leaf_contexts=[context],
            ranked_features_by_leaf={8: [0, 1]},
            random_state=42,
            top_k_features_to_try=2,
            local_feature_subset_size=2,
            local_max_depth=2,
            local_max_leaf_nodes=4,
        )

        single_feature_accuracy = np.mean(single_feature_model.predict(X_train) == y_train)
        multi_feature_accuracy = np.mean(multi_feature_model.predict(X_train) == y_train)

        self.assertLess(single_feature_accuracy, 1.0)
        self.assertEqual(multi_feature_accuracy, 1.0)
        self.assertEqual(multi_feature_model.leaf_expansions[8].feature_ids, [0, 1])


if __name__ == "__main__":
    unittest.main()