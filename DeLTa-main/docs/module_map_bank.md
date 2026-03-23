# DeLTa 模块映射（Bank 迁移验证版）

## 1) Rule Extractor（规则抽取）
- 入口脚本：`run_randforest.py`
- 核心训练与日志产物：`train_model_classical.py`
- 关键输入：`example_datasets/bank/*`
- 关键输出：`results/bank/bank_md20_ml50_tree15.log`

## 2) LLM Interface（Prompt 构造 + 查询 + 解析）
- Prompt 构造：`llm/get_prompts/run_get_prompt.py` + `llm/get_prompts/transform.py`
- 当前模板：`llm/get_prompts/bank.py`
- 在线查询：`llm/query/run_get_answer.py` + `llm/query/get_answer.py`
- 规则解析落地：`llm/get_trees.py`
- 关键输出：
  - `llm/prompts/bank/bank_randfull_md20_ml50_tree15.txt`
  - `llm/answers/bank/*.txt`（当前首次迁移仍为空，受在线阻塞影响）
  - `model/llm_rule/bank.py`

## 3) Gradient Net Trainer（误差修正训练链路）
- 训练入口：`run.py` / `train.py`
- 融合入口：`run_ensemble.py` / `ensemble.py`
- 核心实现：`model/DeLTa.py`、`ensemble.py`、`model/utils.py`
- 关键输出：
  - `results/bank/RF_md20_ml50_tree15_full_cart_0.npy`
  - `results/bank/e_RF_md20_ml50_tree15_bank_cart_0.log`

## 4) 当前首轮验证口径（2026-03-08）
- 数据集：`DELTA_DATASETS=bank`
- 基线参数：`md20/ml50/tree15`
- 训练口径：`run.py --num_answers 1`
- 融合口径：单答案融合，对应 `n_ensemble=0`
- 规则来源：当前优先复用 `model/llm_rule/bank.py` 本地规则文件

## 5) 当前结论
- `run_randforest.py` 已可重新执行并落日志。
- `run_get_prompt.py` 已恢复可执行，能稳定生成 `bank` prompt 文件。
- `train.py` 单答案训练已成功产出 `RF_md20_ml50_tree15_full_cart_0.npy`。
- `ensemble.py` 单答案融合结果：`RF Accuracy MEAN: 0.8986`，`Fused Accuracy MEAN: 0.8991`。
- 在线 `run_get_answer.py` / `get_trees.py` 这一步当前仍受 Cloudflare 阻塞，因此首轮迁移采用“本地规则文件替代在线答案”的闭环验证方式。
