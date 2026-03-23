import unittest
from pathlib import Path


class TestPhase1ExecutionArtifacts(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.delta_root = self.root / "DeLTa-main"

    def test_key_artifacts_exist(self):
        artifacts = [
            self.delta_root / "results" / "bank" / "bank_md20_ml50_tree15.log",
            self.delta_root / "llm" / "prompts" / "bank" / "bank_randfull_md20_ml50_tree15.txt",
            self.delta_root / "results" / "bank" / "e_RF_md20_ml50_tree15_bank_cart_9.log",
        ]
        for artifact in artifacts:
            with self.subTest(artifact=str(artifact)):
                self.assertTrue(artifact.exists())

    def test_ensemble_log_has_metrics(self):
        log_path = self.delta_root / "results" / "bank" / "e_RF_md20_ml50_tree15_bank_cart_9.log"
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("RF Accuracy MEAN", content)
        self.assertIn("Fused Accuracy MEAN", content)

    def test_delta_windows_resource_guard_exists(self):
        file_path = self.delta_root / "model" / "DeLTa.py"
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("try:", content)
        self.assertIn("import resource", content)
        self.assertIn("except Exception", content)


if __name__ == "__main__":
    unittest.main()
