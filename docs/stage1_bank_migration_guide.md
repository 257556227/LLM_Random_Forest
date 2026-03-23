# 阶段一：Bank 数据集第一次迁移验证说明

## 1. 这份文档的定位

这份文档回答的问题只有一个：

> 在 `MNIST` 半通用化之后，为什么第一批非图像迁移验证优先选 `bank`，以及实际落地时应该怎么改。

它不是最终算法文档，也不是实验结论文档，而是“第一次迁移验证”的执行说明。

当前阶段判断也已经明确：
- `docs/stage1_mnist_semigeneral_guide.md` 可以视为已完成；
- 当前主要工作阶段就是这份 `bank` 迁移与调优说明；
- 当 `bank` 这条线完成后，后续会进入“调优 + 完全通用化”，目标是把同一机制扩到更多公开数据集。

---

## 2. 为什么首选 bank

当前仓库里能直接看到的非图像候选数据并不多，现成候选主要是：
- `bank`
- `house_16H_reg`

第一批迁移更推荐 `bank`，原因是：
- 它已经有完整数据目录；
- 它是二分类任务，和当前 `MNIST` 的分类链路更接近；
- 它在 `dataset_config.py` 里已经有现成参数；
- 它已有对应 Prompt 模板 `DeLTa-main/llm/get_prompts/bank.py`；
- 它走的是 `cart` 小模型链路，比回归 + `tabpfn` 的 `house_16H_reg` 更适合做第一轮迁移验证。

一句话：

> `bank` 不是最终最重要的数据集，但它是当前最稳、最便宜、最容易暴露迁移问题的第一站。

---

## 3. Bank 现在已经具备什么

### 3.1 数据产物已在仓库中
目录：`DeLTa-main/example_datasets/bank/`

当前已具备：
- 数值特征：`N_train/N_val/N_test.npy`
- 类别特征：`C_train/C_val/C_test.npy`
- 标签：`y_train/y_val/y_test.npy`
- 元信息：`info.json`

这意味着第一轮迁移不需要先做数据下载脚本，可以直接做流程验证。

### 3.2 配置入口已具备
- 数据配置：`DeLTa-main/dataset_config.py`
- Prompt 模板：`DeLTa-main/llm/get_prompts/bank.py`

也就是说，`bank` 不是从零开始接入，而是一个“已有基础、但还没按当前 MNIST 稳态流程重新走一遍”的对象。

---

## 4. 迁移时要改哪些地方

### 4.1 第一类：先确认，不一定要改
- `DeLTa-main/example_datasets/bank/`
- `DeLTa-main/dataset_config.py`
- `DeLTa-main/llm/get_prompts/bank.py`

先确认这三处是否和当前执行口径一致：
- 数据文件齐不齐；
- `bank` 的参数是否仍想沿用；
- Prompt 是否满足当前输出格式约束。

### 4.2 第二类：大概率要补日志与产物说明
- `DeLTa-main/docs/module_map_mnist.md` 当前只写了 `MNIST`
- `experiments/mnist_experiment_ledger.md` 当前也只记录 `MNIST`

当 `bank` 开始真正执行后，建议新增：
- `DeLTa-main/docs/module_map_bank.md`
- `experiments/bank_experiment_ledger.md`

第一轮迁移的重点不是立刻追高分，而是把 `bank` 的脚本入口、产物路径、日志命名先固化。

### 4.3 第三类：如果规则命名不一致，要优先修这个
迁移最容易踩坑的地方之一不是训练本身，而是：
- LLM 规则文件命名；
- `run.py` 读取的规则名；
- `run_ensemble.py` 读取的结果名。

所以 `bank` 第一次跑时，要重点核对：
- `DeLTa-main/model/llm_rule/bank.py` 是否生成；
- 规则变量名是否和训练入口一致；
- `results/bank/` 下的 `.npy` 和 `.log` 是否按统一口径命名。

---

## 5. 最小执行顺序

推荐严格按这个顺序，不要跳步：

1. `DELTA_DATASETS=bank` 跑 `run_randforest.py`
2. 跑 `run_get_prompt.py`
3. 检查 Prompt 产物是否落到 `llm/prompts/bank/`
4. 再跑 `run_get_answer.py`
5. 跑 `get_trees.py`
6. 确认 `model/llm_rule/bank.py` 是否生成成功
7. 跑 `run.py`
8. 跑 `run_ensemble.py`

中途如果失败，不要同时改三四处。

标准处理方式是：
- 先确定卡在“数据 / Prompt / 查询 / 规则解析 / 训练 / 融合”哪一层；
- 只修当前层；
- 修完就回归，不要顺手大改别的地方。

---

## 6. 第一轮不要急着调哪些参数

第一次做 `bank` 迁移验证时，不建议上来就改：
- Prompt 结构；
- 多套 `eta`；
- 多组 `num_queries`；
- 多组 `n_ensemble`；
- 树深度和树数一起改。

第一轮应优先固定：
- `dataset=bank`
- 现有 `dataset_config.py` 默认参数
- 单一 Prompt 模板（先用 `bank.py`）
- 单一执行链路

先证明“能稳定跑通且产物命名一致”，再谈调优。

---

## 7. 第一次迁移的完成标准

满足下面 5 条，就算 `bank` 第一轮迁移验证完成：
- `run_randforest.py -> run_ensemble.py` 整条链可以执行；
- 每一步都有产物可回查；
- `model/llm_rule/bank.py` 成功落地或有明确替代规则文件；
- `results/bank/` 有统一命名的日志与结果；
- `Update.md` 与实验台账里能讲清这次迁移到底卡在哪或成功在哪。

---

## 8. 这一步和后续新方法的关系

做 `bank` 迁移验证，不是为了继续停留在 DeLTa 基线，而是为了确认：
- 当前这套“半通用化底座”是不是真的能从 `MNIST` 迁到一个 tabular 分类数据集；
- 如果连 `bank` 都迁不稳，后面直接做 `relation / leaf expansion` 只会让问题更难定位。

所以这一轮的价值是：

> 先证明底座可迁移，再把新方法插上去，不然很容易把“工程不稳”误判成“方法没用”。

---

## 9. 当前首轮真实验证结果（2026-03-08）

当前已经完成一轮 `bank` 的真实迁移验证，但口径需要明确写清：
- `run_randforest.py` 已成功重新生成 `results/bank/bank_md20_ml50_tree15.log`；
- `run_get_prompt.py` 已成功重新生成 `llm/prompts/bank/bank_randfull_md20_ml50_tree15.txt`；
- 训练阶段已使用本地规则文件 `model/llm_rule/bank.py` 成功产出 `results/bank/RF_md20_ml50_tree15_full_cart_0.npy`；
- 融合阶段已成功产出 `results/bank/e_RF_md20_ml50_tree15_bank_cart_0.log`；
- 当前单答案口径结果：`RF Accuracy MEAN = 0.8986`，`Fused Accuracy MEAN = 0.8991`。

同时也要明确当前阻塞：
- 在线 `run_get_answer.py` 与 `get_trees.py` 这一步在当前环境中仍受 `codemirror.codes` 的 Cloudflare 阻塞；
- 所以这一轮证明的是“底座可以迁到 bank 并继续训练/融合”，不是“bank 在线答案链已经完全恢复”。

对应沉淀文档：
- 模块映射：`DeLTa-main/docs/module_map_bank.md`
- 实验台账：`experiments/bank_experiment_ledger.md`

## 10. 当前 online relation 迁移结果（2026-03-08）

在完成“本地规则替代在线答案”的首轮闭环后，当前又补跑了一轮 `bank` 上的 leaf expansion online relation 迁移验证：

- `without_llm` 摘要：`DeLTa-main/results/bank/leaf_expansion_without_llm_summary.json`
- `mock_llm` 摘要：`DeLTa-main/results/bank/llm_guided_leaf_expansion_summary.json`
- `online_llm` 摘要：`DeLTa-main/results/bank/llm_guided_leaf_expansion_online_summary.json`
- online 尝试状态：`DeLTa-main/results/bank/bank_leaf_expansion_online_llm_attempt.json`
- 同口径摘要：`DeLTa-main/results/bank/bank_leaf_expansion_same_metric_summary.json`
- 叶子级对比：`DeLTa-main/results/bank/bank_leaf_expansion_online_vs_mock_leaf_compare.json`
- 优先级报告：`DeLTa-main/results/bank/bank_leaf_expansion_optimization_priority.json`
- 策略预览：`DeLTa-main/results/bank/bank_leaf_selection_strategy_preview.json`

当前统一 `test_accuracy` 结果：
- `baseline_rf = 0.8986`
- `without_llm = 0.8886`
- `mock_llm = 0.8890`
- `online_llm = 0.8890`

这说明：
- `bank` 上的 online relation 链已经能真实跑通，不再只是接口位；
- 最新一轮已完成轻量 `bank` 专用 prompt 增强：去掉 tabular 数据上的伪 `pixel` 坐标，并补入 tabular business data / numeric / categorical 指引；
- 随后又补了一轮更强语义 prompt：加入 `task_intro`、`target_outcomes`、`feature_description`，并显式强调 bank 营销/订阅语义；
- 再进一步补入 target-conditioned prompt：加入 `predicted_outcome`、`alternative_outcomes`、`category_values_preview`、`predicted_class_top_categories`，让模型直接看到当前叶子在“订阅/不订阅”上的方向信息和主要类别值；
- 最新又补入 `one_feature_train_accuracy` 与 explicit counterfactual prompt，要求模型对类似 `age` vs `duration` 的取舍做局部反事实检查；
- 最新结果已提升到 `online_llm = 0.8889748977109366`，与 `mock_llm = 0.8889748977109366` 追平；
- 最新叶子级对比显示：`leaf 11` 上 online 已从 `duration` 翻转为 `age`，与 mock 对齐；`leaf 5` 与 `leaf 13` 虽然选特征不同，但局部效果仍打平；
- 随后新增叶子优化优先级分析工具 `DeLTa-main/tools/analyze_bank_leaf_optimization_priority.py`，并生成正式报告 `bank_leaf_expansion_optimization_priority.json`；
- 该报告显示当前 3 个已选叶子在“单特征局部最优”视角下都没有正向 headroom，`estimated_global_accuracy_gain` 全部为 `0.0`；
- 这说明当前主要目标已经不再是“继续改同一批叶子的 prompt 文案”，而是优先判断：
	- 是否要调整候选叶子选择；
	- 是否要扩展局部结构搜索空间；
	- 或者直接把工程重心切到 `adult`，避免在 `bank` 上继续低收益打磨。

## 12. `bank` 叶子选择策略最小实跑结果（2026-03-11）

在上一步确认“当前默认选中的 3 个叶子 headroom 已接近耗尽”后，当前又补了一轮更像工程排查而不是 prompt 微调的动作：

- 新增策略预览工具：`DeLTa-main/tools/analyze_bank_leaf_selection_strategies.py`
- 预览产物：`DeLTa-main/results/bank/bank_leaf_selection_strategy_preview.json`

当前预览结论：
- 默认 `impurity_mass` / `sample_count` 选中的叶子仍是：`[5, 11, 13]`
- `impurity` 策略会切到一组全新的高纯度冲突叶子：`[14, 10, 6]`

基于这组新叶子，当前已补跑最小对照：
- `without_llm + impurity`：`0.8904124737365918`
- `mock_llm + impurity`：`0.8900807254229791`
- `online_llm + impurity`：`0.8900807254229791`

相对于旧默认策略：
- 旧 `without_llm = 0.8886431493973239`
- 旧 `mock_llm = 0.8889748977109366`
- 旧 `online_llm = 0.8889748977109366`

这说明：
- 当前 `bank` 上真正更值得优先改的，确实是候选叶子选择，而不是继续给 `leaf 5 / 11 / 13` 堆更细的 prompt 文案；
- 至少在 `without_llm` 和 `mock_llm` 口径下，`impurity` 已经给出正向提升；
- 当前 `online_llm` 也已在 `impurity` 新叶子组下成功跑通，说明这条工程链路已经完成闭环验证；
- 但 `online_llm` 仍然只是追平 `mock_llm`，还没有在 `bank` 上形成“真实在线 relation 明显超过 mock relation”的新证据；
- 因此如果继续深挖 `bank`，下一优先级应转向更大结构搜索空间、更多候选特征组合，或者切到 `adult` 做正式主对比协议，而不是继续围绕旧叶子写 pairwise prompt。

## 11. 当前推荐的主对比协议（进入调优阶段前先固定）

后续 `bank` 以及更多数据集的正式对比，优先固定为：
- `Deeper RF (depth=4)`
- `depth=3 + without_llm`
- `depth=3 + online_llm`

补充对照：
- `depth=3`：公共底座锚点
- `depth=3 + mock_llm`：sanity check

这里尤其要注意：
- `Deeper RF` 不是 `without_llm`；
- 前者是“整棵树统一多长一层”；
- 后者是“仅在部分候选叶子局部补 1 层”；
- 因此两者必须同时保留，才能区分“传统加深”与“局部结构扩展”到底谁贡献了收益。