# DeLTa 模块映射（Car 正式协议版）

## 1) Rule Extractor（规则抽取）
- 入口脚本：`run_randforest.py`
- 核心训练与日志产物：`train_model_classical.py`
- 预期输入：`example_datasets/car/*`
- 预期输出：`results/car/car_*.log`

## 2) LLM Interface（Prompt 构造 + 查询 + 解析）
- Prompt 构造：`llm/get_prompts/run_get_prompt.py` + `llm/get_prompts/transform.py`
- 当前模板：`llm/get_prompts/car.py`
- 在线查询：`llm/query/run_get_answer.py` + `llm/query/get_answer.py`
- 规则解析落地：`llm/get_trees.py`
- 预期输出：
  - `llm/prompts/car/*.txt`
  - `llm/answers/car/*.txt`
  - `model/llm_rule/car.py`

## 3) Gradient Net Trainer（误差修正训练链路）
- 训练入口：`run.py` / `train.py`
- 融合入口：`run_ensemble.py` / `ensemble.py`
- 核心实现：`model/DeLTa.py`、`ensemble.py`、`model/utils.py`
- 预期输出：
  - `results/car/*.npy`
  - `results/car/*.log`

## 4) Leaf Expansion（局部结构扩展）
- 统一入口：`run_leaf_expansion_mnist.py --dataset car`
- 候选叶子与特征摘要：`model/leaf_expansion/mnist_leaf_selector.py`
- relation prompt：`model/leaf_expansion/mnist_relation_provider.py`
- 局部扩展：`model/leaf_expansion/mnist_leaf_expander.py`
- 预期输出：
  - `results/car/leaf_expansion_without_llm_summary.json`
  - `results/car/llm_guided_leaf_expansion_summary.json`
  - `results/car/car_leaf_expansion_same_metric_summary.json`
  - `results/car/car_leaf_expansion_formal_protocol_summary.json`

## 5) 当前接入判断（2026-03-11）
- `dataset_config.py`、Prompt 模板、rule 文件、数据目录、同口径摘要与正式协议都已存在；
- 当前正式协议结果：`online_llm = 0.8256578947368421`、`deeper_rf = 0.7960526315789473`、`traditional_rf = 0.8174342105263158`、`traditional_delta = 0.8470394736842105`；
- 调优小扫结果：`impurity_mass` 与 `impurity` 等价，`sample_count` 明显更差，增加 `top_k_leaves` 或 `top_k_features_to_try` 也未继续提升；
- 因此 `car` 当前已经可以作为“低维纯类别多分类”正式样板，后续应把经验复制到 `jannis` 或回归路线，而不是继续在 car 上做低收益细磨。
