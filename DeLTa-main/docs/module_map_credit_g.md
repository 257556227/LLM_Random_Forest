# DeLTa 模块映射（Credit-g 正式协议版）

## 1) Rule Extractor（规则抽取）
- 入口脚本：`run_randforest.py`
- 核心训练与日志产物：`train_model_classical.py`
- 预期输入：`example_datasets/credit-g/*`
- 当前已落地产物：
  - `results/credit-g/credit-g_md15_ml20_tree20.log`
  - `results/credit-g/credit-g_md15_ml20_tree20.npy`

## 2) LLM Interface（Prompt 构造 + 查询 + 解析）
- Prompt 构造：`llm/get_prompts/run_get_prompt.py` + `llm/get_prompts/transform.py`
- 当前模板：`llm/get_prompts/credit-g.py`
- 在线查询：`llm/query/run_get_answer.py` + `llm/query/get_answer.py`
- 规则解析落地：`llm/get_trees.py`
- 预期输出：
  - `llm/prompts/credit-g/*.txt`
  - `llm/answers/credit-g/*.txt`
  - `model/llm_rule/credit-g.py`

## 3) Gradient Net Trainer（误差修正训练链路）
- 训练入口：`run.py` / `train.py`
- 融合入口：`run_ensemble.py` / `ensemble.py`
- 核心实现：`model/DeLTa.py`、`ensemble.py`、`model/utils.py`
- 当前已落地产物：
  - `results/credit-g/*.npy`
  - `results/credit-g/*.log`
  - `results/credit-g/RF_md15_ml20_tree20_full_cart_0.npy`
  - `results/credit-g/e_RF_md15_ml20_tree20_credit-g_cart_0.log`

## 4) Leaf Expansion（局部结构扩展）
- 统一入口：`run_leaf_expansion_mnist.py --dataset credit-g`
- 候选叶子与特征摘要：`model/leaf_expansion/mnist_leaf_selector.py`
- relation prompt：`model/leaf_expansion/mnist_relation_provider.py`
- 局部扩展：`model/leaf_expansion/mnist_leaf_expander.py`
- 当前已落地产物：
  - `results/credit-g/leaf_expansion_without_llm_summary.json`
  - `results/credit-g/llm_guided_leaf_expansion_summary.json`
  - `results/credit-g/credit_g_leaf_expansion_same_metric_summary.json`
  - `results/credit-g/credit_g_leaf_expansion_formal_protocol_summary.json`

## 5) 当前接入判断（2026-03-11）
- `dataset_config.py`、Prompt 模板、rule 文件、数据目录与正式协议摘要都已存在；
- 当前主链基线已经补齐：`traditional_rf = 0.7428571428571429`，`traditional_delta = 0.7342857142857143`；
- 当前 `deeper_rf = 0.7314285714285714`，而 `online_llm = 0.7171428571428572`；
- 这说明 `credit-g` 当前不是“还没接上主链”，而是“正式协议已显示 online 暂时落后于统一深度对照和传统主链”，下一步应优先排查 prompt 与候选叶子策略，而不是继续补基础接线。
