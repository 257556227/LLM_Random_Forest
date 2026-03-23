import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "DeLTa-main"


class TestHouse16HRegIntegration(unittest.TestCase):
    def test_house_regression_artifacts_exist(self):
        results_dir = PROJECT_ROOT / "results" / "house_16H_reg"
        self.assertTrue((results_dir / "leaf_expansion_without_llm_summary.json").exists())
        self.assertTrue((results_dir / "llm_guided_leaf_expansion_summary.json").exists())
        self.assertTrue((results_dir / "llm_guided_leaf_expansion_online_summary.json").exists())
        self.assertTrue((results_dir / "house_16H_reg_leaf_expansion_same_metric_summary.json").exists())

        summary = json.loads(
            (results_dir / "house_16H_reg_leaf_expansion_same_metric_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["dataset"], "house_16H_reg")
        self.assertEqual(summary["metric_scope"], "same_test_rmse_eval")
        self.assertIn("prototype_baseline", summary["experiments"])
        self.assertIn("without_llm", summary["experiments"])
        self.assertIn("mock_llm", summary["experiments"])
        self.assertIn("online_llm", summary["experiments"])
        self.assertIn("online_llm_rmse_delta_vs_mock", summary["comparisons"])
        self.assertIn("online_llm_r2_delta_vs_mock", summary["comparisons"])

    def test_house_regression_formal_protocol_exists(self):
        results_dir = PROJECT_ROOT / "results" / "house_16H_reg"
        formal_protocol_file = results_dir / "house_16H_reg_leaf_expansion_formal_protocol_summary.json"
        self.assertTrue(formal_protocol_file.exists())

        formal_protocol = json.loads(formal_protocol_file.read_text(encoding="utf-8"))
        self.assertEqual(formal_protocol["dataset"], "house_16H_reg")
        self.assertTrue(formal_protocol["strict_same_metric_protocol_ready"])
        self.assertEqual(formal_protocol["metric_scope"], "same_test_rmse_eval")
        self.assertIn("deeper_rf", formal_protocol["experiments"])
        self.assertIn("online_llm", formal_protocol["experiments"])
        self.assertIn("traditional_rf", formal_protocol["experiments"])
        self.assertIn("traditional_delta", formal_protocol["experiments"])
        self.assertIn("online_llm_rmse_delta_vs_deeper_rf", formal_protocol["comparisons"])
        self.assertIn("online_llm_rmse_delta_vs_traditional_rf", formal_protocol["comparisons"])
        self.assertIn("online_llm_rmse_delta_vs_traditional_delta", formal_protocol["comparisons"])
        self.assertIn("online_llm_r2_delta_vs_deeper_rf", formal_protocol["comparisons"])
        self.assertIn("online_llm_r2_delta_vs_traditional_rf", formal_protocol["comparisons"])
        self.assertIn("online_llm_r2_delta_vs_traditional_delta", formal_protocol["comparisons"])

    def test_house_regression_docs_exist(self):
        self.assertTrue((ROOT / "experiments" / "house_16H_reg_experiment_ledger.md").exists())
        self.assertTrue((PROJECT_ROOT / "docs" / "module_map_house_16H_reg.md").exists())


if __name__ == "__main__":
    unittest.main()