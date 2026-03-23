import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = ROOT / "DeLTa-main"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

from model import utils as model_utils


class TestTabPFNDeviceSelection(unittest.TestCase):
    def test_prefers_cuda_when_available(self):
        with patch("model.utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(model_utils.get_tabpfn_device(), "cuda")

    def test_falls_back_to_cpu_when_cuda_unavailable(self):
        with patch("model.utils.torch.cuda.is_available", return_value=False):
            self.assertEqual(model_utils.get_tabpfn_device(), "cpu")

    def test_runtime_small_model_falls_back_to_cart_without_cuda(self):
        with patch("model.utils.torch.cuda.is_available", return_value=False):
            self.assertEqual(model_utils.resolve_small_model_for_runtime("tabpfn"), "cart")

    def test_runtime_small_model_keeps_tabpfn_with_cuda(self):
        with patch("model.utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(model_utils.resolve_small_model_for_runtime("tabpfn"), "tabpfn")


if __name__ == "__main__":
    unittest.main()
