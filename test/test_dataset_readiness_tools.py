import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

from tools.audit_dataset_integration_readiness import build_dataset_readiness_report


class TestDatasetReadinessTools(unittest.TestCase):
    def test_build_adult_readiness_report_marks_ready_state(self):
        report = build_dataset_readiness_report("adult")

        self.assertEqual(report["dataset"], "adult")
        self.assertTrue(report["checks"]["dataset_config_exists"])
        self.assertTrue(report["checks"]["prompt_template_exists"])
        self.assertTrue(report["checks"]["llm_rule_exists"])
        self.assertTrue(report["checks"]["results_dir_exists"])
        self.assertTrue(report["checks"]["experiment_ledger_exists"])
        self.assertTrue(report["checks"]["module_map_exists"])
        self.assertTrue(report["checks"]["dataset_dir_exists"])
        self.assertTrue(report["ready_for_first_run"])
        self.assertIn("adult", json.dumps(report, ensure_ascii=False))

    def test_build_credit_g_readiness_report_normalizes_dash_names(self):
        report = build_dataset_readiness_report("credit-g")

        self.assertEqual(report["dataset"], "credit-g")
        self.assertTrue(report["checks"]["dataset_config_exists"])
        self.assertTrue(report["checks"]["prompt_template_exists"])
        self.assertTrue(report["checks"]["llm_rule_exists"])
        self.assertTrue(report["checks"]["precheck_doc_exists"])
        self.assertTrue(report["checks"]["dataset_dir_exists"])
        self.assertTrue(report["checks"]["experiment_ledger_exists"])
        self.assertTrue(report["checks"]["module_map_exists"])
        self.assertTrue(report["ready_for_first_run"])
        self.assertIn("credit-g", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()