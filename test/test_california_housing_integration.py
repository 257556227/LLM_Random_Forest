import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "DeLTa-main"

class TestCaliforniaHousingIntegration(unittest.TestCase):
    def test_california_regression_artifacts_exist(self):
        results_dir = PROJECT_ROOT / "results" / "california_housing"
        self.assertTrue((results_dir / "leaf_expansion_without_llm_summary.json").exists())
        self.assertTrue((results_dir / "llm_guided_leaf_expansion_summary.json").exists())
        self.assertTrue((results_dir / "llm_guided_leaf_expansion_online_summary.json").exists())
        self.assertTrue((results_dir / "california_housing_leaf_expansion_formal_protocol_summary.json").exists())

        summary = json.loads(
            (results_dir / "california_housing_leaf_expansion_formal_protocol_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["dataset"], "california_housing")
        self.assertIn("mainline_rf", summary)
        self.assertIn("mainline_delta", summary)
        self.assertIn("deeper_rf", summary)
        self.assertIn("without_llm", summary)
        self.assertIn("mock_llm", summary)
        self.assertIn("online_llm", summary)

    def test_california_regression_formal_protocol_exists(self):
        results_dir = PROJECT_ROOT / "results" / "california_housing"
        formal_protocol_file = results_dir / "california_housing_leaf_expansion_formal_protocol_summary.json"
        self.assertTrue(formal_protocol_file.exists())

        formal_protocol = json.loads(formal_protocol_file.read_text(encoding="utf-8"))
        self.assertEqual(formal_protocol["dataset"], "california_housing")
        self.assertIn("mainline_rf", formal_protocol)
        self.assertIn("mainline_delta", formal_protocol)
        self.assertIn("deeper_rf", formal_protocol)
        self.assertIn("online_llm", formal_protocol)
