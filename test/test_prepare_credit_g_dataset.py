import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

from tools.prepare_credit_g_dataset import build_credit_g_info


class TestPrepareCreditGDataset(unittest.TestCase):
    def test_build_credit_g_info_contains_expected_schema(self):
        info = build_credit_g_info(train_size=500, val_size=150, test_size=350)

        self.assertEqual(info["task_type"], "binclass")
        self.assertEqual(info["num_classes"], 2)
        self.assertEqual(info["n_num_features"], 7)
        self.assertEqual(info["n_cat_features"], 13)
        self.assertIn("feature_intro", info)
        self.assertIn("target_intro", info)
        self.assertIn("duration", info["feature_intro"]["num"])
        self.assertIn("checking_status", info["feature_intro"]["cat"])
        self.assertIn("good", info["target_intro"])
        self.assertIn("bad", info["target_intro"])


if __name__ == "__main__":
    unittest.main()
