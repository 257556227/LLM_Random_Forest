# DeLTa 模块映射（Jannis 正式协议版）

## 1) Rule Extractor（规则抽取）
- 入口脚本：`run_randforest.py`
- 核心训练与日志产物：`train_model_classical.py`
- 预期输入：`example_datasets/jannis/*`
- 当前已落地产物：
  - `results/jannis/jannis_md20_ml50_tree20.log`
  - `results/jannis/jannis_md20_ml50_tree20.npy`

## 2) LLM Interface（Prompt 构造 + 查询 + 解析）
- Prompt 构造：`llm/get_prompts/run_get_prompt.py` + `llm/get_prompts/transform.py`
- 当前模板：`llm/get_prompts/jannis.py`
- 在线查询：`llm/query/run_get_answer.py` + `llm/query/get_answer.py`
- 规则解析落地：`llm/get_trees.py`
- 预期输出：
  - `llm/prompts/jannis/*.txt`
  - `llm/answers/jannis/*.txt`
  - `model/llm_rule/jannis.py`

## 3) Gradient Net Trainer（误差修正训练链路）
- 训练入口：`run.py` / `train.py`
- 融合入口：`run_ensemble.py` / `ensemble.py`
- 核心实现：`model/DeLTa.py`、`ensemble.py`、`model/utils.py`
- 当前已落地产物：
  - `results/jannis/RF_md20_ml50_tree20_full_cart_*.npy`
  - `results/jannis/RF_md20_ml50_tree20_jannis_shotfull_*_cart.log`
  - `results/jannis/e_RF_md20_ml50_tree20_jannis_cart_0.log`

## 4) Leaf Expansion（局部结构扩展）
- 统一入口：`run_leaf_expansion_mnist.py --dataset jannis`
- 候选叶子与特征摘要：`model/leaf_expansion/mnist_leaf_selector.py`
- relation prompt：`model/leaf_expansion/mnist_relation_provider.py`
- 局部扩展：`model/leaf_expansion/mnist_leaf_expander.py`
- 预期产物：
  - `results/jannis/leaf_expansion_without_llm_summary.json`
  - `results/jannis/llm_guided_leaf_expansion_summary.json`
  - `results/jannis/llm_guided_leaf_expansion_online_summary.json`
  - `results/jannis/jannis_leaf_expansion_same_metric_summary.json`
  - `results/jannis/jannis_leaf_expansion_formal_protocol_summary.json`

## 5) 当前接入判断（2026-03-11）
- `dataset_config.py`、Prompt 模板、rule 文件、数据目录、三组原型和正式协议都已存在；
- 当前正式协议结果：`online_llm = 0.5814175673254911`、`deeper_rf = 0.6324117752433271`、`traditional_rf = 0.64877291455186`、`traditional_delta = 0.6574908938914432`；
- 调优小扫结果：`impurity` 更差，`sample_count`、`top_k_leaves=4`、`top_k_features_to_try=4` 都只追平默认，没有继续提升；
- 因此 `jannis` 当前已经可以作为“高维弱语义多分类”正式样板，结论是：online 语义引导对原型有效，但当前还不足以超过统一深度对照和传统主链。