# DeLTa 模块映射（MNIST执行版）

## 1) Rule Extractor（规则抽取）
- 入口脚本：`run_randforest.py`
- 核心训练与日志产物：`train_model_classical.py`
- 关键输入：`example_datasets/mnist/*`
- 关键输出：`results/mnist/mnist_md8_ml20_tree3.log`

## 2) LLM Interface（Prompt构造 + 查询 + 解析）
- Prompt构造：`llm/get_prompts/run_get_prompt.py` + `llm/get_prompts/transform.py`
- 稳定模板：`llm/get_prompts/mnist_stable.py`
- 在线查询：`llm/query/run_get_answer.py` + `llm/query/get_answer.py`
- 规则解析落地：`llm/get_trees.py`
- 关键输出：
  - `llm/prompts/mnist/mnist_randfull_md8_ml20_tree3.txt`
  - `llm/answers/mnist/mnist_randfull_md8_ml20_tree3_0..9.txt`
  - `model/llm_rule/mnist.py`

## 3) Gradient Net Trainer（误差修正训练链路）
- 训练入口：`run.py`
- 融合入口：`run_ensemble.py`
- 核心实现：`model/DeLTa.py`、`ensemble.py`、`model/utils.py`
- 关键输出：
  - `results/mnist/RF_md8_ml20_tree3_full_cart_0..9.npy`
  - `results/mnist/e_RF_md8_ml20_tree3_mnist_cart_9.log`

## 4) 当前稳定参数（高效果）
- 数据集：`DELTA_DATASETS=mnist`
- 在线查询：
  - `num_queries=10`
  - `request_interval=2`
  - `max_completion_tokens=2200`
  - `normalize_tree_output=1`
- 训练/融合：
  - `run.py --num_answers 10`
  - `run_ensemble.py --n_ensemble 9 --eta 0.2`

## 5) 最近结果（2026-02-19）
- `Fused Accuracy MEAN: 0.9260`
- `Fused AUC MEAN: 0.99236`
