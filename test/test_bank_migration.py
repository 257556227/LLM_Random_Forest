import unittest
from pathlib import Path


class TestBankMigration(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.delta_root = self.root / "DeLTa-main"

    def test_bank_migration_docs_exist(self):
        guide = self.root / "docs" / "stage1_bank_migration_guide.md"
        module_map = self.delta_root / "docs" / "module_map_bank.md"
        ledger = self.root / "experiments" / "bank_experiment_ledger.md"

        self.assertTrue(guide.exists())
        self.assertTrue(module_map.exists())
        self.assertTrue(ledger.exists())

        module_map_content = module_map.read_text(encoding="utf-8", errors="ignore")
        ledger_content = ledger.read_text(encoding="utf-8", errors="ignore")

        self.assertIn("DeLTa 模块映射（Bank 迁移验证版）", module_map_content)
        self.assertIn("Gradient Net Trainer", module_map_content)
        self.assertIn("在线阻塞", module_map_content)
        self.assertIn("Bank 实验台账（第一次迁移验证）", ledger_content)
        self.assertIn("本地规则文件替代在线答案", ledger_content)
        self.assertIn("Cloudflare", ledger_content)

    def test_bank_migration_artifacts_exist(self):
        result_dir = self.delta_root / "results" / "bank"
        prompt_file = self.delta_root / "llm" / "prompts" / "bank" / "bank_randfull_md20_ml50_tree15.txt"
        rule_file = self.delta_root / "model" / "llm_rule" / "bank.py"
        baseline_log = result_dir / "bank_md20_ml50_tree15.log"
        train_output = result_dir / "RF_md20_ml50_tree15_full_cart_0.npy"
        ensemble_log = result_dir / "e_RF_md20_ml50_tree15_bank_cart_0.log"
        without_summary = result_dir / "leaf_expansion_without_llm_summary.json"
        mock_summary = result_dir / "llm_guided_leaf_expansion_summary.json"
        online_summary = result_dir / "llm_guided_leaf_expansion_online_summary.json"
        online_attempt = result_dir / "bank_leaf_expansion_online_llm_attempt.json"
        same_metric_summary = result_dir / "bank_leaf_expansion_same_metric_summary.json"
        same_metric_impurity_summary = result_dir / "bank_leaf_expansion_same_metric_impurity.json"
        leaf_compare_summary = result_dir / "bank_leaf_expansion_online_vs_mock_leaf_compare.json"
        priority_summary = result_dir / "bank_leaf_expansion_optimization_priority.json"
        strategy_summary = result_dir / "bank_leaf_selection_strategy_preview.json"
        online_attempt_impurity = result_dir / "bank_leaf_expansion_online_llm_attempt_impurity.json"
        online_impurity_summary = result_dir / "llm_guided_leaf_expansion_online_impurity_summary.json"
        online_attempt_tool = self.delta_root / "tools" / "attempt_bank_online_llm_same_metric.py"
        same_metric_tool = self.delta_root / "tools" / "evaluate_bank_leaf_expansion_same_metric.py"
        leaf_compare_tool = self.delta_root / "tools" / "compare_bank_leaf_expansion_online_vs_mock.py"
        priority_tool = self.delta_root / "tools" / "analyze_bank_leaf_optimization_priority.py"
        strategy_tool = self.delta_root / "tools" / "analyze_bank_leaf_selection_strategies.py"

        self.assertTrue(prompt_file.exists())
        self.assertTrue(rule_file.exists())
        self.assertTrue(baseline_log.exists())
        self.assertTrue(train_output.exists())
        self.assertTrue(ensemble_log.exists())
        self.assertTrue(without_summary.exists())
        self.assertTrue(mock_summary.exists())
        self.assertTrue(online_summary.exists())
        self.assertTrue(online_attempt.exists())
        self.assertTrue(same_metric_summary.exists())
        self.assertTrue(same_metric_impurity_summary.exists())
        self.assertTrue(leaf_compare_summary.exists())
        self.assertTrue(priority_summary.exists())
        self.assertTrue(strategy_summary.exists())
        self.assertTrue(online_attempt_impurity.exists())
        self.assertTrue(online_impurity_summary.exists())
        self.assertTrue(online_attempt_tool.exists())
        self.assertTrue(same_metric_tool.exists())
        self.assertTrue(leaf_compare_tool.exists())
        self.assertTrue(priority_tool.exists())
        self.assertTrue(strategy_tool.exists())

        try:
            ensemble_content = ensemble_log.read_text(encoding="utf-8", errors="ignore")
            if "RF Accuracy MEAN" not in ensemble_content:
                ensemble_content = ensemble_log.read_text(encoding="utf-16", errors="ignore")
        except UnicodeError:
            ensemble_content = ensemble_log.read_text(encoding="utf-16", errors="ignore")
        same_metric_content = same_metric_summary.read_text(encoding="utf-8", errors="ignore")
        same_metric_impurity_content = same_metric_impurity_summary.read_text(encoding="utf-8", errors="ignore")
        online_summary_content = online_summary.read_text(encoding="utf-8", errors="ignore")
        online_impurity_summary_content = online_impurity_summary.read_text(encoding="utf-8", errors="ignore")
        leaf_compare_content = leaf_compare_summary.read_text(encoding="utf-8", errors="ignore")
        priority_content = priority_summary.read_text(encoding="utf-8", errors="ignore")
        strategy_content = strategy_summary.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("RF Accuracy MEAN", ensemble_content)
        self.assertIn("Fused Accuracy MEAN", ensemble_content)
        self.assertIn('"online_llm"', same_metric_content)
        self.assertIn('"online_llm_minus_mock_llm"', same_metric_content)
        self.assertIn('"online_llm"', same_metric_impurity_content)
        self.assertIn('"leaf_selection_strategy": "impurity"', same_metric_impurity_content)
        self.assertIn("tabular business data", online_summary_content)
        self.assertIn("tabular business data", online_impurity_summary_content)
        self.assertIn("term deposit", online_summary_content)
        self.assertIn("term deposit", online_impurity_summary_content)
        self.assertIn('"leaf_comparisons"', leaf_compare_content)
        self.assertIn('"mock_selected_feature_name"', leaf_compare_content)
        self.assertIn('"online_selected_feature_name"', leaf_compare_content)
        self.assertIn('"priority_targets"', priority_content)
        self.assertIn('"estimated_global_accuracy_gain"', priority_content)
        self.assertIn('"recommended_prompt_focus"', priority_content)
        self.assertIn('"strategies"', strategy_content)
        self.assertIn('"impurity_mass"', strategy_content)
        self.assertIn('"sample_count"', strategy_content)
        self.assertIn("feature_name=balance", online_summary_content)
        self.assertNotIn("pixel=(0,1)", online_summary_content)


if __name__ == "__main__":
    unittest.main()
