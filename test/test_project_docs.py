import unittest
from pathlib import Path


class TestProjectDocs(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_known_info_doc_exists(self):
        doc = self.root / "docs" / "project_known_info.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("当前项目的真实边界", content)
        self.assertIn("LLM-Guided Leaf Expansion", content)
        self.assertIn("统一实验协议", content)
        self.assertIn("负结果同样重要", content)
        self.assertIn("完全通用化目标", content)
        self.assertIn("Deeper RF / without_llm / online_llm", content)

    def test_stage1_semigeneral_guide_exists(self):
        doc = self.root / "docs" / "stage1_mnist_semigeneral_guide.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("MNIST 半通用化操作说明", content)
        self.assertIn("当前最重要的几个板块", content)
        self.assertIn("如果后面要换数据集", content)
        self.assertIn("参数应该先调哪里", content)
        self.assertIn("顶级程序员视角的任务拆解", content)
        self.assertIn("现在就该做的动作顺序", content)
        self.assertIn("当前完成标准与阻塞标准", content)
        self.assertIn("为什么现在还要做 `bank` 验证", content)
        self.assertIn("可以开始 `MNIST` 版 `LLM-Guided Leaf Expansion Forest` 原型", content)
        self.assertIn("五类入口速查表", content)
        self.assertIn("入口 -> 常见产物 -> 排查位置", content)
        self.assertIn("当前有没有偏离原计划", content)
        self.assertIn("受控偏移", content)
        self.assertIn("当前还需要继续什么", content)
        self.assertIn("统一四组协议", content)
        self.assertIn("online_llm_attempt", content)
        self.assertTrue(
            ("Cloudflare" in content)
            or ("401" in content)
            or ("invalid_api_key" in content)
            or ("status = success" in content)
        )
        self.assertNotIn("前置条件表”还没写出来", content)
        self.assertNotIn("还没被写成明确操作表", content)

    def test_bank_migration_guide_exists(self):
        doc = self.root / "docs" / "stage1_bank_migration_guide.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("Bank 数据集第一次迁移验证说明", content)
        self.assertIn("为什么首选 bank", content)
        self.assertIn("迁移时要改哪些地方", content)
        self.assertIn("最小执行顺序", content)
        self.assertIn("当前推荐的主对比协议", content)
        self.assertIn("Deeper RF", content)

    def test_mnist_leaf_expansion_prototype_doc_exists(self):
        doc = self.root / "docs" / "mnist_llm_guided_leaf_expansion_prototype.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("LLM-Guided Leaf Expansion Forest 原型说明", content)
        self.assertIn("这一版不做什么", content)
        self.assertIn("顶级程序员视角的原型拆解", content)
        self.assertIn("第一版完成标准", content)
        self.assertIn("online_llm", content)
        self.assertIn("正式在线 LLM relation", content)
        self.assertIn("mnist_leaf_expansion_protocol_summary.json", content)
        self.assertIn("run_mnist_four_way_protocol.py", content)
        self.assertIn("mnist_leaf_expansion_same_metric_summary.json", content)
        self.assertIn("mnist_leaf_expansion_online_llm_attempt.json", content)
        self.assertIn("local_openai_config.json", content)
        self.assertIn("8 个数据集", content)
        self.assertIn("generalization_scaling_lessons.md", content)
        self.assertTrue(
            ("Cloudflare" in content)
            or ("401" in content)
            or ("invalid_api_key" in content)
            or ("status = success" in content)
        )

    def test_generalization_lessons_doc_exists(self):
        doc = self.root / "docs" / "generalization_scaling_lessons.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("调优与完全通用化推进记录", content)
        self.assertIn("8", content)
        self.assertIn("Deeper RF", content)
        self.assertIn("online_llm", content)
        self.assertIn("adult", content)
        self.assertIn("接入作战卡", content)
        self.assertIn("当前预检文档索引", content)
        self.assertIn("jannis_integration_precheck.md", content)
        self.assertIn("car_integration_precheck.md", content)
        self.assertIn("house_16H_reg_integration_precheck.md", content)
        self.assertIn("california_housing_integration_precheck.md", content)

    def test_dataset_generalization_methodology_doc_exists(self):
        doc = self.root / "docs" / "dataset_generalization_methodology.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("数据集通用化方法论", content)
        self.assertIn("adult / credit-g / car", content)
        self.assertIn("Deeper RF", content)
        self.assertIn("without_llm", content)
        self.assertIn("online_llm", content)
        self.assertIn("jannis", content)
        self.assertIn("house_16H_reg", content)
        self.assertIn("先分类数据集类型", content)

    def test_adult_precheck_doc_exists(self):
        doc = self.root / "docs" / "adult_integration_precheck.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("adult 数据集接入前检查清单", content)
        self.assertIn("当前核查结论", content)
        self.assertIn("example_datasets/adult", content)
        self.assertIn("run_leaf_expansion_mnist.py", content)

    def test_credit_g_precheck_doc_exists(self):
        doc = self.root / "docs" / "credit_g_integration_precheck.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("credit-g 数据集接入前检查清单", content)
        self.assertIn("当前核查结论", content)
        self.assertIn("example_datasets/credit-g", content)
        self.assertIn("run_leaf_expansion_mnist.py", content)

    def test_jannis_precheck_doc_exists(self):
        doc = self.root / "docs" / "jannis_integration_precheck.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("jannis 数据集接入前检查清单", content)
        self.assertIn("当前核查结论", content)
        self.assertIn("example_datasets/jannis", content)
        self.assertIn("高维多分类表格数据", content)
        self.assertIn("语义通常弱于 `adult` / `credit-g`", content)

    def test_car_precheck_doc_exists(self):
        doc = self.root / "docs" / "car_integration_precheck.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("car 数据集接入前检查清单", content)
        self.assertIn("当前核查结论", content)
        self.assertIn("example_datasets/car", content)
        self.assertIn("低维、纯类别型、多分类数据", content)

    def test_house_16h_reg_precheck_doc_exists(self):
        doc = self.root / "docs" / "house_16H_reg_integration_precheck.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("house_16H_reg 数据集接入前检查清单", content)
        self.assertIn("当前核查结论", content)
        self.assertIn("example_datasets/house_16H_reg", content)
        self.assertIn("task_type = regression", content)

    def test_california_housing_precheck_doc_exists(self):
        doc = self.root / "docs" / "california_housing_integration_precheck.md"
        self.assertTrue(doc.exists())
        content = doc.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("california_housing 数据集接入前检查清单", content)
        self.assertIn("当前核查结论", content)
        self.assertIn("example_datasets/california_housing", content)
        self.assertIn("run_leaf_expansion_mnist.py", content)

    def test_readme_mentions_stage1_guides(self):
        readme = self.root / "docs" / "readme.md"
        self.assertTrue(readme.exists())
        content = readme.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("docs/stage1_mnist_semigeneral_guide.md", content)
        self.assertIn("docs/stage1_bank_migration_guide.md", content)
        self.assertIn("docs/mnist_llm_guided_leaf_expansion_prototype.md", content)
        self.assertIn("mnist_leaf_expansion_protocol_summary.json", content)
        self.assertIn("run_mnist_four_way_protocol.py", content)
        self.assertIn("mnist_leaf_expansion_same_metric_summary.json", content)
        self.assertIn("mnist_leaf_expansion_online_llm_attempt.json", content)
        self.assertIn("mnist_leaf_expansion_oracle_analysis.json", content)
        self.assertIn("generalization_scaling_lessons.md", content)
        self.assertIn("dataset_generalization_methodology.md", content)
        self.assertIn("adult_integration_precheck.md", content)
        self.assertIn("credit_g_integration_precheck.md", content)
        self.assertIn("jannis_integration_precheck.md", content)
        self.assertIn("car_integration_precheck.md", content)
        self.assertIn("house_16H_reg_integration_precheck.md", content)
        self.assertIn("california_housing_integration_precheck.md", content)
        self.assertTrue(
            ("Cloudflare" in content)
            or ("401" in content)
            or ("invalid_api_key" in content)
            or ("success" in content)
        )

    def test_copilot_instructions_mentions_local_openai_config(self):
        instructions = self.root / ".github" / "copilot-instructions.md"
        self.assertTrue(instructions.exists())
        content = instructions.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("docs/readme.md", content)
        self.assertIn("docs/lookme.md", content)
        self.assertIn("local_openai_config.json", content)
        self.assertIn("不要删除该自动加载逻辑", content)
        self.assertIn("除非用户明确要求，否则不要修改其中内容", content)
        self.assertIn("generalization_scaling_lessons.md", content)
        self.assertIn("adult_integration_precheck.md", content)
        self.assertIn("credit_g_integration_precheck.md", content)
        self.assertIn("jannis_integration_precheck.md", content)
        self.assertIn("car_integration_precheck.md", content)
        self.assertIn("house_16H_reg_integration_precheck.md", content)
        self.assertIn("california_housing_integration_precheck.md", content)


if __name__ == "__main__":
    unittest.main()