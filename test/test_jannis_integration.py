import json
import unittest
from pathlib import Path


class TestJannisIntegration(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.delta_root = self.root / "DeLTa-main"

    def test_jannis_data_and_artifacts_exist(self):
        dataset_dir = self.delta_root / "example_datasets" / "jannis"
        results_dir = self.delta_root / "results" / "jannis"
        readiness_file = results_dir / "jannis_integration_readiness.json"
        without_file = results_dir / "leaf_expansion_without_llm_summary.json"
        mock_file = results_dir / "llm_guided_leaf_expansion_summary.json"
        online_file = results_dir / "llm_guided_leaf_expansion_online_summary.json"
        same_metric_file = results_dir / "jannis_leaf_expansion_same_metric_summary.json"
        formal_protocol_file = results_dir / "jannis_leaf_expansion_formal_protocol_summary.json"

        for file_name in [
            "N_train.npy",
            "N_val.npy",
            "N_test.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test.npy",
            "info.json",
        ]:
            self.assertTrue((dataset_dir / file_name).exists(), file_name)

        self.assertTrue(readiness_file.exists())
        self.assertTrue(without_file.exists())
        self.assertTrue(mock_file.exists())
        self.assertTrue(online_file.exists())
        self.assertTrue(same_metric_file.exists())
        self.assertTrue(formal_protocol_file.exists())

        readiness = json.loads(readiness_file.read_text(encoding="utf-8"))
        summary = json.loads(same_metric_file.read_text(encoding="utf-8"))
        formal_protocol = json.loads(formal_protocol_file.read_text(encoding="utf-8"))

        self.assertTrue(readiness["ready_for_first_run"])
        self.assertEqual(summary["dataset"], "jannis")
        self.assertIn("online_llm", summary["experiments"])
        self.assertIn("online_llm_minus_mock_llm", summary["comparisons"])
        self.assertEqual(formal_protocol["dataset"], "jannis")
        self.assertTrue(formal_protocol["strict_same_metric_protocol_ready"])
        self.assertIn("traditional_rf", formal_protocol["experiments"])
        self.assertIn("traditional_delta", formal_protocol["experiments"])
        self.assertIn("deeper_rf", formal_protocol["experiments"])
        self.assertIn("online_llm", formal_protocol["experiments"])
        self.assertIn("online_llm_minus_deeper_rf", formal_protocol["comparisons"])
        self.assertIn("online_llm_minus_traditional_rf", formal_protocol["comparisons"])
        self.assertIn("online_llm_minus_traditional_delta", formal_protocol["comparisons"])


if __name__ == "__main__":
    unittest.main()