import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

from tools.prepare_adult_dataset import build_adult_info


class TestPrepareAdultDataset(unittest.TestCase):
    def test_build_adult_info_contains_expected_schema(self):
        info = build_adult_info(train_size=26048, val_size=6513, test_size=16281)

        self.assertEqual(info["task_type"], "binclass")
        self.assertEqual(info["num_classes"], 2)
        self.assertEqual(info["n_num_features"], 6)
        self.assertEqual(info["n_cat_features"], 8)
        self.assertIn("feature_intro", info)
        self.assertIn("target_intro", info)
        self.assertIn("age", info["feature_intro"]["num"])
        self.assertIn("workclass", info["feature_intro"]["cat"])
        self.assertIn(">50K", " ".join(info["target_intro"].keys()))


if __name__ == "__main__":
    unittest.main()