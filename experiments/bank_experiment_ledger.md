# Bank 实验台账（第一次迁移验证）

| 日期 | 配置 | 规则来源 | 在线答案数 | n_ensemble | eta | RF Acc | Fused Acc | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-03-08 | md20/ml50/tree15 | `model/llm_rule/bank.py` | 0（在线阻塞） | 0 | 0.2 | 0.8986 | 0.8991 | 首轮迁移验证，采用本地规则文件替代在线答案 |

## 首轮真实执行结果
- 基线日志：`DeLTa-main/results/bank/bank_md20_ml50_tree15.log`
- Prompt 产物：`DeLTa-main/llm/prompts/bank/bank_randfull_md20_ml50_tree15.txt`
- 规则文件：`DeLTa-main/model/llm_rule/bank.py`
- 训练产物：`DeLTa-main/results/bank/RF_md20_ml50_tree15_full_cart_0.npy`
- 融合日志：`DeLTa-main/results/bank/e_RF_md20_ml50_tree15_bank_cart_0.log`

## 首轮执行命令要点
- 基线：`DELTA_DATASETS=bank python run_randforest.py`
- Prompt：`DELTA_DATASETS=bank python llm/get_prompts/run_get_prompt.py`
- 训练：`python train.py --task_type full --small_model cart --classify_rule model.llm_rule.bank.tree_full_md20_ml50_0 --dataset bank --dataset_path example_datasets --seed_num 1 --model_type DeLTa --gpu 0 --cat_policy ordinal --save_npy results/bank/RF_md20_ml50_tree15_full_cart_0`
- 融合：`python ensemble.py --small_model cart --num_classes 2 --n_ensemble 0 --md 20 --ml 50 --tree 15 --dataset bank`

## 当前阻塞
- 在线 `run_get_answer.py` 和后续 `get_trees.py` 仍受 `codemirror.codes` 的 Cloudflare 页面拦截。
- 因此这轮验证回答的是“底座是否能迁移到 bank 并继续训练/融合”，还不是“online answers 在 bank 上是否稳定可得”。

## 下一步
- 在可用网络/代理条件下补跑 `run_get_answer.py -> get_trees.py`。
- 若在线链恢复，再把 `num_answers=10` / `n_ensemble=9` 的稳定口径迁过去，形成与 `MNIST` 更接近的对照协议。

## leaf expansion online relation 迁移验证（2026-03-08）
- 执行入口：`conda run -n py310 python run_leaf_expansion_mnist.py --dataset bank --mode without_llm --save_summary`
- 执行入口：`conda run -n py310 python run_leaf_expansion_mnist.py --dataset bank --mode llm_guided --relation_source mock_llm --save_summary`
- 执行入口：`conda run -n py310 python tools/attempt_bank_online_llm_same_metric.py`
- 汇总入口：`conda run -n py310 python tools/evaluate_bank_leaf_expansion_same_metric.py`
- 结果文件：
	- `DeLTa-main/results/bank/leaf_expansion_without_llm_summary.json`
	- `DeLTa-main/results/bank/llm_guided_leaf_expansion_summary.json`
	- `DeLTa-main/results/bank/llm_guided_leaf_expansion_online_summary.json`
	- `DeLTa-main/results/bank/bank_leaf_expansion_online_llm_attempt.json`
	- `DeLTa-main/results/bank/bank_leaf_expansion_same_metric_summary.json`
- 当前同口径结果：
	- `baseline_rf = 0.8986`
	- `without_llm = 0.8886`
	- `mock_llm = 0.8890`
	- `online_llm = 0.8886`
	- `online_llm_minus_mock_llm = -0.0003`
- 当前结论：
	- `bank` 上的 online relation 已能真实执行成功；
	- 但当前增益没有像 `MNIST` 那样保住，online 与 without_llm 持平，略低于 mock_llm；
	- 最新一轮已完成轻量 `bank` 专用 prompt 增强：去掉伪 `pixel` 坐标，并补入 tabular business data / numeric / categorical 指引；
	- 这更像是 prompt 语义与候选特征描述还不够贴合 `bank`，而不是迁移接线问题。

## 更强语义 prompt + online-vs-mock 叶子级对比（2026-03-08）
- 执行入口：`conda run -n py310 python run_leaf_expansion_mnist.py --dataset bank --mode llm_guided --relation_source mock_llm --save_summary --guided_output results/bank/llm_guided_leaf_expansion_summary.json`
- 执行入口：`conda run -n py310 python tools/attempt_bank_online_llm_same_metric.py`
- 执行入口：`conda run -n py310 python tools/compare_bank_leaf_expansion_online_vs_mock.py`
- 结果文件：
	- `DeLTa-main/results/bank/llm_guided_leaf_expansion_summary.json`
	- `DeLTa-main/results/bank/llm_guided_leaf_expansion_online_summary.json`
	- `DeLTa-main/results/bank/bank_leaf_expansion_online_vs_mock_leaf_compare.json`
- 当前 prompt 增量：
	- 新增 `task_intro`
	- 新增 `target_outcomes`
	- 新增 `feature_description`
	- 新增 bank 营销 / term deposit 领域指引
	- 新增 `predicted_outcome` / `alternative_outcomes`
	- 新增 `category_values_preview` / `predicted_class_top_categories` / `other_class_top_categories`
- 当前叶子级结论：
	- `leaf 5`：online 选 `pdays`，mock 选 `job`，局部 `test_accuracy` 打平
	- `leaf 11`：online 选 `duration`，mock 选 `age`，mock 局部 `test_accuracy` 高 `0.0059`
	- `leaf 13`：online 选 `poutcome`，mock 选 `age`，局部 `test_accuracy` 打平
- 当前判断：
	- 更强 bank 语义 prompt 已经改变了 online 的叶子内排序，但主要损失集中在 `leaf 11`；
	- 当前 target-conditioned prompt 和类别值解释已经接入，但 `leaf 11` 仍未翻转；
	- 下一步最值得补的是更细粒度类别值解释或 explicit counterfactual prompt，而不是继续改接线。

## counterfactual prompt：online 追平 mock（2026-03-08）
- 执行入口：`conda run -n py310 python tools/attempt_bank_online_llm_same_metric.py`
- 执行入口：`conda run -n py310 python tools/evaluate_bank_leaf_expansion_same_metric.py`
- 执行入口：`conda run -n py310 python tools/compare_bank_leaf_expansion_online_vs_mock.py`
- 变更文件：
	- `DeLTa-main/model/leaf_expansion/mnist_leaf_selector.py`
	- `DeLTa-main/model/leaf_expansion/mnist_relation_provider.py`
	- `DeLTa-main/results/bank/llm_guided_leaf_expansion_online_summary.json`
	- `DeLTa-main/results/bank/bank_leaf_expansion_same_metric_summary.json`
	- `DeLTa-main/results/bank/bank_leaf_expansion_online_vs_mock_leaf_compare.json`
- 当前 prompt 增量：
	- 新增 `one_feature_train_accuracy`
	- 新增 explicit counterfactual 指令，要求比较类似 `age` vs `duration` 的局部取舍
	- 保留已有 `predicted_outcome` / `alternative_outcomes` / 类别值提示
- 当前同口径结果：
	- `baseline_rf = 0.8986`
	- `without_llm = 0.8886`
	- `mock_llm = 0.8889748977109366`
	- `online_llm = 0.8889748977109366`
	- `online_llm_minus_mock_llm = 0.0`
- 当前叶子级结论：
	- `leaf 11`：online 已从 `duration` 翻转为 `age`，与 mock 对齐
	- `leaf 5` 与 `leaf 13`：仍有排序差异，但局部 `test_accuracy` 打平
- 当前判断：
	- counterfactual prompt 已证明 online 在 `bank` 上可以追平 `mock_llm`，不是“真模型天然不如假排序”；
	- 下一步应继续筛查还有没有其他叶子值得做 pairwise / leaf-specific prompt，争取让 online 超过 mock。

## 叶子优化优先级报告（2026-03-10）
- 执行入口：`conda run -n py310 python tools/analyze_bank_leaf_optimization_priority.py`
- 结果文件：
	- `DeLTa-main/results/bank/bank_leaf_expansion_optimization_priority.json`
- 当前结论：
	- 当前 `selected_leaf_ids = [5, 11, 13]` 的 `estimated_global_accuracy_gain` 全部为 `0.0`；
	- `leaf 5`、`leaf 11`、`leaf 13` 在当前“单特征局部最优”视角下都没有额外正向 headroom；
	- 这意味着继续围绕同一批叶子写更细的 pairwise / counterfactual prompt，大概率已经进入低收益区。
- 当前判断：
	- 如果还要让 `bank online_llm` 超过 `mock_llm`，更值得优先改的是候选叶子选择策略或局部结构搜索空间，而不是继续调当前这 3 个叶子的措辞；
	- 从排期角度看，也可以考虑先把 `bank` 视为“prompt 侧已基本收口”，把更多精力转到 `adult` 的第一轮真实接入。
