import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

from tools.prepare_car_dataset import build_car_info


class TestPrepareCarDataset(unittest.TestCase):
    def test_build_car_info_contains_expected_schema(self):
        info = build_car_info(train_size=864, val_size=256, test_size=608)

        self.assertEqual(info["task_type"], "multiclass")
        self.assertEqual(info["num_classes"], 4)
        self.assertEqual(info["n_num_features"], 0)
        self.assertEqual(info["n_cat_features"], 6)
        self.assertIn("feature_intro", info)
        self.assertIn("target_intro", info)
        self.assertIn("buying", info["feature_intro"]["cat"])
        self.assertIn("safety", info["feature_intro"]["cat"])
        self.assertIn("unacc", info["target_intro"])
        self.assertIn("vgood", info["target_intro"])


if __name__ == "__main__":
    unittest.main()