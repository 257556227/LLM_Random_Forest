import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
MODULE_PATH = DELTA_ROOT / "tools" / "prepare_jannis_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_jannis_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_jannis_info = MODULE.build_jannis_info


class TestPrepareJannisDataset(unittest.TestCase):
    def test_build_jannis_info_contains_expected_schema(self):
        info = build_jannis_info(train_size=53588, val_size=13398, test_size=16747)

        self.assertEqual(info["task_type"], "multiclass")
        self.assertEqual(info["num_classes"], 4)
        self.assertEqual(info["n_num_features"], 54)
        self.assertEqual(info["n_cat_features"], 0)
        self.assertIn("feature_intro", info)
        self.assertIn("target_intro", info)
        self.assertIn("V1", info["feature_intro"]["num"])
        self.assertIn("V54", info["feature_intro"]["num"])
        self.assertIn("0", info["target_intro"])
        self.assertIn("3", info["target_intro"])


if __name__ == "__main__":
    unittest.main()