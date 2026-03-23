# DeLTa 模块映射（Adult 接入准备版）

## 1) Rule Extractor（规则抽取）
- 入口脚本：`run_randforest.py`
- 核心训练与日志产物：`train_model_classical.py`
- 预期输入：`example_datasets/adult/*`
- 预期输出：`results/adult/adult_*.log`

## 2) LLM Interface（Prompt 构造 + 查询 + 解析）
- Prompt 构造：`llm/get_prompts/run_get_prompt.py` + `llm/get_prompts/transform.py`
- 当前模板：`llm/get_prompts/adult.py`
- 在线查询：`llm/query/run_get_answer.py` + `llm/query/get_answer.py`
- 规则解析落地：`llm/get_trees.py`
- 预期输出：
  - `llm/prompts/adult/*.txt`
  - `llm/answers/adult/*.txt`
  - `model/llm_rule/adult.py`

## 3) Gradient Net Trainer（误差修正训练链路）
- 训练入口：`run.py` / `train.py`
- 融合入口：`run_ensemble.py` / `ensemble.py`
- 核心实现：`model/DeLTa.py`、`ensemble.py`、`model/utils.py`
- 预期输出：
  - `results/adult/*.npy`
  - `results/adult/*.log`

## 4) Leaf Expansion（局部结构扩展）
- 统一入口：`run_leaf_expansion_mnist.py --dataset adult`
- 候选叶子与特征摘要：`model/leaf_expansion/mnist_leaf_selector.py`
- relation prompt：`model/leaf_expansion/mnist_relation_provider.py`
- 局部扩展：`model/leaf_expansion/mnist_leaf_expander.py`
- 预期输出：
  - `results/adult/leaf_expansion_without_llm_summary.json`
  - `results/adult/llm_guided_leaf_expansion_summary.json`
  - `results/adult/*same_metric_summary.json`

## 5) 当前接入判断（2026-03-11）
- `dataset_config.py`、Prompt 模板、rule 文件已存在；
- `example_datasets/adult/` 尚未落地，因此当前还不能开始第一轮真实运行；
- 下一步优先级不是继续解释方法，而是补数据目录与 `info.json`，然后按 `without_llm -> mock_llm -> online_llm` 顺序推进。
