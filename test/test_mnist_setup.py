import unittest
from pathlib import Path


class TestMnistSetup(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.delta_root = self.root / "DeLTa-main"

    def test_mnist_dataset_files_exist(self):
        dataset_dir = self.delta_root / "example_datasets" / "mnist"
        expected_files = [
            "N_train.npy",
            "N_val.npy",
            "N_test.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test.npy",
            "info.json",
        ]
        for name in expected_files:
            with self.subTest(name=name):
                self.assertTrue((dataset_dir / name).exists())

    def test_mnist_prompt_and_config_exist(self):
        prompt_file = self.delta_root / "llm" / "get_prompts" / "mnist.py"
        stable_prompt_file = self.delta_root / "llm" / "get_prompts" / "mnist_stable.py"
        ab_prompt_file = self.delta_root / "llm" / "get_prompts" / "mnist_ab_b.py"
        aligned_prompt_file = self.delta_root / "llm" / "get_prompts" / "mnist_prompt12_aligned.py"
        config_file = self.delta_root / "dataset_config.py"
        rule_file = self.delta_root / "model" / "llm_rule" / "mnist.py"

        self.assertTrue(prompt_file.exists())
        self.assertTrue(stable_prompt_file.exists())
        self.assertTrue(ab_prompt_file.exists())
        self.assertTrue(aligned_prompt_file.exists())
        self.assertTrue(config_file.exists())
        self.assertTrue(rule_file.exists())

        prompt_content = prompt_file.read_text(encoding="utf-8", errors="ignore")
        config_content = config_file.read_text(encoding="utf-8", errors="ignore")
        rule_content = rule_file.read_text(encoding="utf-8", errors="ignore")

        self.assertIn('"name": "mnist"', prompt_content)
        self.assertIn("'mnist'", config_content)
        self.assertIn("tree_full_md8_ml20_0", rule_content)
        self.assertIn("tree_full_md8_ml20_1", rule_content)
        self.assertIn("tree_full_md8_ml20_9", rule_content)

        stable_prompt_content = stable_prompt_file.read_text(encoding="utf-8", errors="ignore")
        ab_prompt_content = ab_prompt_file.read_text(encoding="utf-8", errors="ignore")
        aligned_prompt_content = aligned_prompt_file.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("ONLY one Python code block", stable_prompt_content)
        self.assertIn("Return ONLY one Python code block", ab_prompt_content)
        self.assertIn("inter-feature associations", aligned_prompt_content)

    def test_env_dataset_switch_supported(self):
        scripts = [
            self.delta_root / "run_randforest.py",
            self.delta_root / "run.py",
            self.delta_root / "run_ensemble.py",
            self.delta_root / "llm" / "get_prompts" / "run_get_prompt.py",
            self.delta_root / "llm" / "query" / "run_get_answer.py",
            self.delta_root / "llm" / "get_trees.py",
        ]
        for script in scripts:
            with self.subTest(script=str(script)):
                content = script.read_text(encoding="utf-8", errors="ignore")
                self.assertIn("DELTA_DATASETS", content)

    def test_prompt_template_env_supported(self):
        run_prompt_file = self.delta_root / "llm" / "get_prompts" / "run_get_prompt.py"
        content = run_prompt_file.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("DELTA_PROMPT_TEMPLATE", content)

    def test_run_supports_configurable_answer_count(self):
        run_file = self.delta_root / "run.py"
        ensemble_file = self.delta_root / "run_ensemble.py"
        run_content = run_file.read_text(encoding="utf-8", errors="ignore")
        ensemble_content = ensemble_file.read_text(encoding="utf-8", errors="ignore")

        self.assertIn("--num_answers", run_content)
        self.assertIn("DELTA_NUM_ANSWERS", run_content)
        self.assertIn("--n_ensemble", ensemble_content)
        self.assertIn("DELTA_N_ENSEMBLE", ensemble_content)
        self.assertIn("--eta", ensemble_content)

    def test_query_uses_local_or_env_api_key(self):
        query_file = self.delta_root / "llm" / "query" / "get_answer.py"
        helper_file = self.delta_root / "llm" / "openai_local_config.py"
        local_config_file = self.delta_root / "local_openai_config.json"
        content = query_file.read_text(encoding="utf-8", errors="ignore")
        helper_content = helper_file.read_text(encoding="utf-8", errors="ignore")
        local_config_content = local_config_file.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("OPENAI_API_KEY", content)
        self.assertIn("get_openai_settings", content)
        self.assertTrue(helper_file.exists())
        self.assertTrue(local_config_file.exists())
        self.assertIn("OPENAI_BASE_URL", helper_content)
        self.assertIn("OPENAI_MODEL", helper_content)
        self.assertIn('"OPENAI_API_KEY":', local_config_content)
        self.assertIn('"OPENAI_BASE_URL": "https://api.openai.com/v1"', local_config_content)
        self.assertIn('"OPENAI_MODEL": "gpt-4o"', local_config_content)
        self.assertNotIn('"<sk-xxx>"', content)
        self.assertNotIn("sk-proj-", content)
        self.assertNotIn('"OPENAI_API_KEY": "<sk-xxx>"', local_config_content)

    def test_query_supports_split_and_rate_params(self):
        query_file = self.delta_root / "llm" / "query" / "get_answer.py"
        run_query_file = self.delta_root / "llm" / "query" / "run_get_answer.py"
        content = query_file.read_text(encoding="utf-8", errors="ignore")
        run_content = run_query_file.read_text(encoding="utf-8", errors="ignore")

        self.assertIn("split_rules_max_chars", content)
        self.assertIn("split_rules_overlap_chars", content)
        self.assertIn("request_interval", content)
        self.assertIn("max_completion_tokens", content)
        self.assertIn("normalize_tree_output", content)

        self.assertIn("--split_rules_max_chars", run_content)
        self.assertIn("--split_rules_overlap_chars", run_content)
        self.assertIn("--request_interval", run_content)
        self.assertIn("--max_completion_tokens", run_content)
        self.assertIn("--normalize_tree_output", run_content)

    def test_mnist_reduced_profile_artifacts(self):
        result_dir = self.delta_root / "results" / "mnist"
        self.assertTrue((result_dir / "mnist_md8_ml20_tree3.npy").exists())
        self.assertTrue((result_dir / "mnist_md8_ml20_tree3_train.npy").exists())
        ensemble_log = result_dir / "e_RF_md8_ml20_tree3_mnist_cart_9.log"
        self.assertTrue(ensemble_log.exists())

        content = ensemble_log.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("RF Accuracy MEAN", content)
        self.assertIn("Fused Accuracy MEAN", content)

    def test_docs_ledger_and_check_script_exist(self):
        module_map = self.delta_root / "docs" / "module_map_mnist.md"
        flow_doc = self.delta_root / "docs" / "overall_model_flow.md"
        check_script = self.delta_root / "tools" / "check_mnist_pipeline.py"
        ledger = self.root / "experiments" / "mnist_experiment_ledger.md"
        lookme = self.root / "docs" / "lookme.md"

        self.assertTrue(module_map.exists())
        self.assertTrue(flow_doc.exists())
        self.assertTrue(check_script.exists())
        self.assertTrue(ledger.exists())
        self.assertTrue(lookme.exists())

        module_map_content = module_map.read_text(encoding="utf-8", errors="ignore")
        flow_doc_content = flow_doc.read_text(encoding="utf-8", errors="ignore")
        ledger_content = ledger.read_text(encoding="utf-8", errors="ignore")
        lookme_content = lookme.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("Rule Extractor", module_map_content)
        self.assertIn("LLM Interface", module_map_content)
        self.assertIn("Gradient Net Trainer", module_map_content)
        self.assertIn("flowchart TD", flow_doc_content)
        self.assertIn("特征抽取与预处理", flow_doc_content)
        self.assertIn("规则解析与映射", flow_doc_content)
        self.assertIn("融合评估与指标对照", flow_doc_content)
        self.assertNotIn(".py", flow_doc_content)
        self.assertNotIn(".md", flow_doc_content)
        self.assertIn("稳定复现实验命令要点", ledger_content)
        self.assertIn("Prompt A=mnist_stable.py", ledger_content)
        self.assertIn("Prompt B=mnist_ab_b.py", ledger_content)
        self.assertIn("oracle 单特征上界分析", ledger_content)
        self.assertTrue(
            ("一页化执行总结" in lookme_content) or ("执行总结（一步一步照着做）" in lookme_content)
        )
        self.assertTrue(
            ("网络可达后，完成 Prompt-C" in lookme_content)
            or ("Prompt-C（Prompt1/2 对齐版）同口径评估并补齐对照结论" in lookme_content)
        )

    def test_mnist_leaf_expansion_prototype_code_exists(self):
        run_script = self.delta_root / "run_leaf_expansion_mnist.py"
        leaf_expansion_dir = self.delta_root / "model" / "leaf_expansion"
        relation_json_file = self.delta_root / "configs" / "leaf_expansion" / "mnist_mock_relation.json"
        selector_file = leaf_expansion_dir / "mnist_leaf_selector.py"
        provider_file = leaf_expansion_dir / "mnist_relation_provider.py"
        expander_file = leaf_expansion_dir / "mnist_leaf_expander.py"
        init_file = leaf_expansion_dir / "__init__.py"
        summary_file = self.delta_root / "results" / "mnist" / "leaf_expansion_without_llm_summary.json"
        guided_summary_file = self.delta_root / "results" / "mnist" / "llm_guided_leaf_expansion_summary.json"
        guided_json_summary_file = self.delta_root / "results" / "mnist" / "llm_guided_leaf_expansion_json_summary.json"

        self.assertTrue(run_script.exists())
        self.assertTrue(leaf_expansion_dir.exists())
        self.assertTrue(relation_json_file.exists())
        self.assertTrue(selector_file.exists())
        self.assertTrue(provider_file.exists())
        self.assertTrue(expander_file.exists())
        self.assertTrue(init_file.exists())
        self.assertTrue(summary_file.exists())
        self.assertTrue(guided_summary_file.exists())
        self.assertTrue(guided_json_summary_file.exists())

        run_content = run_script.read_text(encoding="utf-8", errors="ignore")
        selector_content = selector_file.read_text(encoding="utf-8", errors="ignore")
        provider_content = provider_file.read_text(encoding="utf-8", errors="ignore")
        expander_content = expander_file.read_text(encoding="utf-8", errors="ignore")
        relation_json_content = relation_json_file.read_text(encoding="utf-8", errors="ignore")
        summary_content = summary_file.read_text(encoding="utf-8", errors="ignore")
        guided_summary_content = guided_summary_file.read_text(encoding="utf-8", errors="ignore")
        guided_json_summary_content = guided_json_summary_file.read_text(encoding="utf-8", errors="ignore")

        self.assertIn("Leaf Expansion without LLM", run_content)
        self.assertIn("--top_k_leaves", run_content)
        self.assertIn("--relation_source", run_content)
        self.assertIn("--mode", run_content)
        self.assertIn("online_llm", run_content)
        self.assertIn("--llm_max_retries", run_content)
        self.assertIn("--llm_retry_delay", run_content)
        self.assertIn("--llm_request_interval", run_content)
        self.assertIn("select_candidate_leaves", selector_content)
        self.assertIn("build_leaf_contexts", selector_content)
        self.assertIn("rank_features_without_llm", provider_content)
        self.assertIn("build_llm_leaf_prompt", provider_content)
        self.assertIn("rank_features_with_mock_llm", provider_content)
        self.assertIn("load_ranked_features_from_json", provider_content)
        self.assertIn("query_ranked_features_with_llm", provider_content)
        self.assertIn("validate_ranked_features_response", provider_content)
        self.assertIn("parse_ranked_features_response", provider_content)
        self.assertIn("get_openai_settings", provider_content)
        self.assertIn('"10"', relation_json_content)
        self.assertIn('"8"', relation_json_content)
        self.assertIn("ExpandedLeafModel", expander_content)
        self.assertIn("fit_leaf_expansion", expander_content)
        self.assertIn("top_k_features_to_try", expander_content)
        self.assertIn('"prototype": "Leaf Expansion without LLM"', summary_content)
        self.assertIn('"baseline_accuracy"', summary_content)
        self.assertIn('"expanded_accuracy"', summary_content)
        self.assertIn('"prototype": "LLM-Guided Leaf Expansion"', guided_summary_content)
        self.assertIn('"relation_source"', guided_summary_content)
        self.assertIn('"leaf_prompts"', guided_summary_content)
        self.assertIn('"selected_expansion_features_by_leaf"', guided_summary_content)
        self.assertIn('"relation_source": "json"', guided_json_summary_content)

    def test_mnist_four_way_protocol_artifacts_exist(self):
        tool_file = self.delta_root / "tools" / "build_mnist_protocol_summary.py"
        runner_file = self.delta_root / "tools" / "run_mnist_four_way_protocol.py"
        protocol_config = self.delta_root / "configs" / "leaf_expansion" / "mnist_four_way_protocol.json"
        summary_file = self.delta_root / "results" / "mnist" / "mnist_leaf_expansion_protocol_summary.json"
        same_metric_tool = self.delta_root / "tools" / "evaluate_mnist_same_metric_protocol.py"
        same_metric_file = self.delta_root / "results" / "mnist" / "mnist_leaf_expansion_same_metric_summary.json"
        online_attempt_tool = self.delta_root / "tools" / "attempt_mnist_online_llm_same_metric.py"
        online_attempt_file = self.delta_root / "results" / "mnist" / "mnist_leaf_expansion_online_llm_attempt.json"
        oracle_tool = self.delta_root / "tools" / "analyze_mnist_leaf_oracle.py"
        oracle_file = self.delta_root / "results" / "mnist" / "mnist_leaf_expansion_oracle_analysis.json"
        repeat_tool = self.delta_root / "tools" / "repeat_mnist_online_llm_same_metric.py"
        repeat_file = self.delta_root / "results" / "mnist" / "mnist_online_llm_repeatability_summary.json"

        self.assertTrue(tool_file.exists())
        self.assertTrue(runner_file.exists())
        self.assertTrue(protocol_config.exists())
        self.assertTrue(summary_file.exists())
        self.assertTrue(same_metric_tool.exists())
        self.assertTrue(same_metric_file.exists())
        self.assertTrue(online_attempt_tool.exists())
        self.assertTrue(online_attempt_file.exists())
        self.assertTrue(oracle_tool.exists())
        self.assertTrue(oracle_file.exists())
        self.assertTrue(repeat_tool.exists())
        self.assertTrue(repeat_file.exists())

        tool_content = tool_file.read_text(encoding="utf-8", errors="ignore")
        runner_content = runner_file.read_text(encoding="utf-8", errors="ignore")
        config_content = protocol_config.read_text(encoding="utf-8", errors="ignore")
        summary_content = summary_file.read_text(encoding="utf-8", errors="ignore")
        same_metric_content = same_metric_file.read_text(encoding="utf-8", errors="ignore")
        online_attempt_content = online_attempt_file.read_text(encoding="utf-8", errors="ignore")
        oracle_content = oracle_file.read_text(encoding="utf-8", errors="ignore")
        repeat_content = repeat_file.read_text(encoding="utf-8", errors="ignore")

        self.assertIn("baseline_rf", tool_content)
        self.assertIn("deeper_rf", tool_content)
        self.assertIn("without_llm", tool_content)
        self.assertIn("llm_guided", tool_content)
        self.assertIn("--dry_run", runner_content)
        self.assertIn("--stage", runner_content)
        self.assertIn("baseline_rf", config_content)
        self.assertIn("deeper_rf", config_content)
        self.assertIn("without_llm", config_content)
        self.assertIn("llm_guided", config_content)
        self.assertIn("build_summary", config_content)
        self.assertIn("same_metric_eval", config_content)
        self.assertIn("online_llm_same_metric_attempt", config_content)

        self.assertIn('"protocol_name": "mnist_leaf_expansion_four_way_v1"', summary_content)
        self.assertIn('"baseline_rf"', summary_content)
        self.assertIn('"deeper_rf"', summary_content)
        self.assertIn('"without_llm"', summary_content)
        self.assertIn('"llm_guided"', summary_content)
        self.assertIn('"metric_scope"', summary_content)
        self.assertIn('"strict_same_metric_protocol_ready": false', summary_content)

        self.assertIn('"protocol_name": "mnist_leaf_expansion_same_metric_v1"', same_metric_content)
        self.assertIn('"metric_scope": "same_test_accuracy_eval"', same_metric_content)
        self.assertIn('"baseline_rf"', same_metric_content)
        self.assertIn('"deeper_rf"', same_metric_content)
        self.assertIn('"without_llm"', same_metric_content)
        self.assertIn('"llm_guided"', same_metric_content)
        self.assertIn('"strict_same_metric_protocol_ready": true', same_metric_content)
        self.assertIn('"online_llm_attempt"', same_metric_content)
        self.assertIn('"online_llm"', same_metric_content)
        self.assertIn('"status"', online_attempt_content)
        self.assertIn('"relation_source": "online_llm"', online_attempt_content)
        self.assertIn('"focus_leaf_ids"', oracle_content)
        self.assertIn('"8"', oracle_content)
        self.assertIn('"9"', oracle_content)
        self.assertIn('"oracle_best_feature"', oracle_content)
        self.assertIn('"repeats"', repeat_content)
        self.assertIn('"success_count"', repeat_content)
        self.assertIn('"wins_over_mock_count"', repeat_content)
        self.assertIn('"repeats": 5', repeat_content)
        repeat_tool_content = repeat_tool.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("--repeats", repeat_tool_content)
        self.assertTrue(
            ('"status": "blocked"' in same_metric_content)
            or ('"status": "failed"' in same_metric_content)
            or ('"status": "success"' in same_metric_content)
        )
        self.assertTrue(
            ('"status": "blocked"' in online_attempt_content)
            or ('"status": "failed"' in online_attempt_content)
            or ('"status": "success"' in online_attempt_content)
        )
        self.assertTrue(
            ('Cloudflare' in online_attempt_content)
            or ('401' in online_attempt_content)
            or ('invalid_api_key' in online_attempt_content)
            or ('"status": "success"' in online_attempt_content)
        )


if __name__ == "__main__":
    unittest.main()
