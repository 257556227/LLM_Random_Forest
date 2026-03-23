# 这个项目是做什么的

我们先用传统树模型拿到一批“基础规则”，再让大模型帮我们总结更好的“决策策略”，最后把这些策略和基线模型融合，看看准确率能不能提升。

你可以把它理解成：
- 传统模型负责“打底”
- 大模型负责“提炼规则”
- 融合模块负责“取长补短”

---

## 你会完成什么

按现在已经跑通的流程，你最终会得到：
1. 一套可复现的 MNIST 全链路实验（从数据到结果）。
2. Prompt A/B/C 的同口径对照结果。
3. 可回查的日志、结果文件、更新记录。
4. 一个覆盖不少于 `8` 个公开数据集的统一扩展框架，并能明确区分哪些数据集已经超过 `deeper_rf`、哪些当前还做不到。

---

## 当前主线（先看这个）

当前主线已经从“继续围绕单个数据集追 `deeper_rf`”调整为：

1. **先把 8 个候选数据集都补到完整闭环**：`readiness + 三组原型 + 正式协议 + 台账 + 测试`；
2. **优先只做高收益动作**：补数据层/readiness、换候选叶子、补局部质量信号、扩大局部结构搜索；
3. **如果某个数据集做完高收益动作仍明显低于传统主对照**，先正式记录为反例样板，后面再统一讨论是否继续冲线。

当前已经验证过的经验是：
- `adult`、`car` 已经能超过 `deeper_rf`；
- `credit-g`、`jannis` 目前还不行；
- `jannis` 上“双特征局部树”比“更深单特征树”更有价值，但仍不足以翻过 `deeper_rf`；
- 因此现阶段最重要的问题不再是“单点能不能立刻翻盘”，而是“8 个数据集的正例/反例边界到底在哪里”。

---

## 文件说明（先知道这些就够了）

- `docs/readme.md`：项目主入口文档（当前这份）。
- `docs/lookme.md`：总执行清单（按阶段勾选）。
- `Update.md`：每次改动和实验结果的更新日志。
- `docs/stage1_mnist_semigeneral_guide.md`：当前阶段操作说明，专门回答“改数据去哪里改、调参数先看哪里”。
- `docs/stage1_bank_migration_guide.md`：第一次非图像迁移验证说明，当前推荐先用 `bank`。
- `docs/mnist_llm_guided_leaf_expansion_prototype.md`：`MNIST` 上的 `LLM-Guided Leaf Expansion Forest` 原型说明。
- `docs/generalization_scaling_lessons.md`：调优与完全通用化推进记录，专门沉淀多数据集扩展过程中的适配经验与失败教训。
- `docs/dataset_generalization_methodology.md`：把已完成样板固化成后续迁移 `jannis / 回归` 的统一方法论，专门回答“新数据集到底该怎么接”。
- `docs/adult_integration_precheck.md`：`adult` 数据集接入前检查清单，用于真正开工前确认数据层、输出路径和同口径摘要缺口。
- `docs/credit_g_integration_precheck.md`：`credit-g` 数据集接入前检查清单，用于第二个强语义表格数据的工程预检。
- `docs/jannis_integration_precheck.md`：`jannis` 数据集接入前检查清单，用于高维弱语义多分类数据的工程预检。
- `docs/car_integration_precheck.md`：`car` 数据集接入前检查清单，用于低维纯类别多分类数据的工程预检。
- `docs/house_16H_reg_integration_precheck.md`：`house_16H_reg` 数据集接入前检查清单，用于第一个回归样板的工程预检。
- `docs/california_housing_integration_precheck.md`：`california_housing` 数据集接入前检查清单，用于第二个回归数据集的工程预检。
- `DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`：`MNIST` 四组协议的统一库存摘要（当前明确标注并非严格同口径最终结论）。
- `DeLTa-main/tools/run_mnist_four_way_protocol.py`：`MNIST` 四组协议统一 runner，支持 `--dry_run` 预览与按阶段执行。
- `DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`：`MNIST` 四组协议统一 `test_accuracy` 同口径摘要。
- `DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json`：真实 `online_llm` 同口径尝试结果或阻塞记录。
- `DeLTa-main/results/mnist/mnist_online_llm_repeatability_summary.json`：`MNIST` online relation 小重复实验摘要，用来判断 `+0.0019` 是否稳定。
- `DeLTa-main/results/mnist/mnist_leaf_expansion_oracle_analysis.json`：叶子级 oracle 单特征上界分析，当前重点看 `leaf 8/9`。
- `DeLTa-main/local_openai_config.json`：本地 OpenAI 私有配置文件，支持自动加载 `OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL`，默认已预填官方 URL 与 `gpt-4o`。
- `DeLTa-main/docs/module_map_bank.md`：`bank` 第一次迁移验证的模块映射与关键产物定位。
- `DeLTa-main/results/bank/bank_leaf_expansion_same_metric_summary.json`：`bank` 上 `baseline_rf / without_llm / mock_llm / online_llm` 的同口径摘要。
- `DeLTa-main/results/bank/bank_leaf_expansion_online_vs_mock_leaf_compare.json`：`bank` 上 `online_llm` 与 `mock_llm` 的叶子级对比结果。
- `DeLTa-main/results/bank/bank_leaf_expansion_optimization_priority.json`：`bank` 上继续优化 online 的叶子优先级报告，用来判断下一步该继续修 prompt，还是该改候选叶子选择。
- `DeLTa-main/results/bank/bank_leaf_selection_strategy_preview.json`：`bank` 上不同候选叶子选择策略的预览结果，当前已确认 `impurity` 会切到一组新叶子。
- `DeLTa-main/results/bank/bank_leaf_expansion_same_metric_impurity.json`：`bank` 上切到 `impurity` 叶子组后的同口径摘要，当前 `online_llm = mock_llm = 0.8900807254229791`，`without_llm = 0.8904124737365918`。
- `DeLTa-main/results/adult/adult_integration_readiness.json`：`adult` 的接入就绪审计结果，用来确认当前缺的是数据层还是代码层。
- `DeLTa-main/results/adult/adult_leaf_expansion_same_metric_summary.json`：`adult` 第一轮原型同口径摘要，当前已出现 `online_llm > without_llm > mock_llm`。
- `DeLTa-main/results/adult/adult_leaf_expansion_formal_protocol_summary.json`：`adult` 正式主对比协议摘要，当前 `online_llm` 已超过 `Deeper RF`，但仍低于传统 `RF / DeLTa` 主链。
- `DeLTa-main/tools/prepare_adult_dataset.py`：`adult` 数据准备脚本，负责真实生成 `example_datasets/adult/` 与 `info.json`。
- `DeLTa-main/tools/evaluate_adult_formal_protocol.py`：`adult` 正式协议汇总脚本，统一汇总 `Deeper RF / without_llm / online_llm / 传统 RF / 传统 DeLTa`。
- `experiments/adult_experiment_ledger.md`：`adult` 第一轮真实接入台账，记录数据准备、原型结果和后续主协议缺口。
- `DeLTa-main/results/credit-g/credit-g_integration_readiness.json`：`credit-g` 的接入就绪审计结果，当前数据目录与台账骨架已落地。
- `DeLTa-main/results/credit-g/credit_g_leaf_expansion_same_metric_summary.json`：`credit-g` 第一轮原型同口径摘要，当前 `mock_llm` 把 `without_llm` 的下滑拉回基线，但 `online_llm` 还没超过 `mock_llm`。
- `DeLTa-main/results/credit-g/credit_g_leaf_expansion_formal_protocol_summary.json`：`credit-g` 正式主对比协议摘要，当前 `online_llm` 低于 `Deeper RF / 传统 RF / 传统 DeLTa`。
- `DeLTa-main/results/car/car_integration_readiness.json`：`car` 的接入就绪审计结果，当前数据目录与台账已经齐全。
- `DeLTa-main/results/car/car_leaf_expansion_same_metric_summary.json`：`car` 首轮同口径摘要，当前 `online_llm` 已明显高于 `mock_llm / without_llm`。
- `DeLTa-main/results/car/car_leaf_expansion_formal_protocol_summary.json`：`car` 正式主对比协议摘要，当前 `online_llm` 已超过 `Deeper RF` 和传统 `RF`，但仍低于传统 `DeLTa`。
- `DeLTa-main/results/jannis/jannis_integration_readiness.json`：`jannis` 的接入就绪审计结果，当前数据目录与台账已经齐全。
- `DeLTa-main/results/jannis/jannis_leaf_expansion_same_metric_summary.json`：`jannis` 首轮同口径摘要，当前 `online_llm` 已明显高于 `mock_llm / without_llm / baseline`。
- `DeLTa-main/results/jannis/jannis_leaf_expansion_formal_protocol_summary.json`：`jannis` 正式主对比协议摘要，当前 `online_llm` 对原型显著有效，但仍低于 `Deeper RF` 和传统 `RF / DeLTa`。
- `DeLTa-main/results/house_16H_reg/house_16H_reg_leaf_expansion_same_metric_summary.json`：`house_16H_reg` 第一轮回归同口径摘要，当前 `online_llm` 已优于 `mock_llm / without_llm / baseline`。
- `DeLTa-main/results/house_16H_reg/house_16H_reg_leaf_expansion_formal_protocol_summary.json`：`house_16H_reg` 正式回归主对比协议摘要，当前 `online_llm` 仍低于 `Deeper RF / 传统 RF / 传统 DeLTa(cart回退)`。
- `DeLTa-main/tools/prepare_credit_g_dataset.py`：`credit-g` 数据准备脚本，负责真实生成 `example_datasets/credit-g/` 与 `info.json`。
- `DeLTa-main/tools/evaluate_credit_g_leaf_expansion_same_metric.py`：`credit-g` 首轮 leaf expansion 同口径摘要脚本。
- `DeLTa-main/tools/evaluate_credit_g_formal_protocol.py`：`credit-g` 正式协议汇总脚本，统一汇总 `Deeper RF / without_llm / online_llm / 传统 RF / 传统 DeLTa`。
- `DeLTa-main/tools/prepare_car_dataset.py`：`car` 数据准备脚本，负责真实生成 `example_datasets/car/` 与 `info.json`。
- `DeLTa-main/tools/evaluate_car_leaf_expansion_same_metric.py`：`car` 首轮 leaf expansion 同口径摘要脚本。
- `DeLTa-main/tools/evaluate_car_formal_protocol.py`：`car` 正式协议汇总脚本，统一汇总 `Deeper RF / without_llm / online_llm / 传统 RF / 传统 DeLTa`。
- `DeLTa-main/tools/prepare_jannis_dataset.py`：`jannis` 数据准备脚本，负责真实生成 `example_datasets/jannis/` 与 `info.json`。
- `DeLTa-main/tools/evaluate_jannis_leaf_expansion_same_metric.py`：`jannis` 首轮 leaf expansion 同口径摘要脚本。
- `DeLTa-main/tools/evaluate_jannis_formal_protocol.py`：`jannis` 正式协议汇总脚本，统一汇总 `Deeper RF / without_llm / online_llm / 传统 RF / 传统 DeLTa`。
- `DeLTa-main/tools/evaluate_house_16H_reg_leaf_expansion_same_metric.py`：`house_16H_reg` 第一轮回归 leaf expansion 同口径摘要脚本。
- `DeLTa-main/tools/evaluate_house_16H_reg_formal_protocol.py`：`house_16H_reg` 正式回归协议汇总脚本，统一汇总 `Deeper RF / without_llm / online_llm / 传统 RF / 传统 DeLTa`，并记录 `tabpfn -> cart` 回退信息。
- `experiments/car_experiment_ledger.md`：`car` 正式协议台账，记录默认配置、调优小扫与当前最优收口结论。
- `experiments/jannis_experiment_ledger.md`：`jannis` 正式协议台账，记录高维弱语义多分类样板的默认配置、调优小扫与当前结论。
- `experiments/house_16H_reg_experiment_ledger.md`：`house_16H_reg` 正式回归样板台账，记录三组原型、正式协议与当前反例判断。
- `experiments/credit_g_experiment_ledger.md`：`credit-g` 第一轮真实接入台账，记录数据准备、原型结果和后续正式协议缺口。
- `DeLTa-main/`：主工程代码（训练、查询、融合都在这里）。
- `experiments/mnist_experiment_ledger.md`：MNIST 实验台账（A/B/C对照在这里看）。
- `experiments/bank_experiment_ledger.md`：`bank` 第一次迁移验证台账（当前是本地规则替代在线答案的闭环）。
- `test/`：单元测试目录。

### 当前文档收敛建议

- 核心长期保留：`docs/readme.md`、`docs/lookme.md`、`Update.md`、`docs/stage1_mnist_semigeneral_guide.md`、`docs/mnist_llm_guided_leaf_expansion_prototype.md`。
- 支撑型文档按主题保留：`docs/stage1_bank_migration_guide.md`、`docs/project_known_info.md`、`docs/generalization_scaling_lessons.md`、`docs/adult_integration_precheck.md`、`docs/credit_g_integration_precheck.md`、`docs/jannis_integration_precheck.md`、`docs/car_integration_precheck.md`、`docs/house_16H_reg_integration_precheck.md`、`docs/california_housing_integration_precheck.md`、`DeLTa-main/docs/module_map_mnist.md`。
- 历史过程文档已转入 `docs/archive/`：`docs/archive/lookme_origin.md`、`docs/archive/lookme整体步骤.md`、`docs/archive/draft.md`，主入口后续不再继续扩写这些旧文档。

---

## 一步一步跑

> 默认在 Windows，且已有 conda 环境 `py310`。

### 第 1 步：准备环境
目标：确认 Python 和依赖可用。

需要做两件事：
1. 激活 `py310`。
2. 确认测试能跑（这是最稳的健康检查）。

---

### 第 2 步：准备数据（MNIST）
目标：确保 `example_datasets/mnist` 下的数据齐全。

你会看到这些文件：
- `N_train.npy / N_val.npy / N_test.npy`
- `y_train.npy / y_val.npy / y_test.npy`
- `info.json`

有这些文件，说明数据层没问题。

---

### 第 3 步：先跑传统基线（规则抽取）
目标：先拿到传统树模型的规则与基线结果。

这一步的作用是“打底”：
- 先有基线，后面才能看 Prompt 优化有没有真实收益。

---

### 第 4 步：生成 Prompt
目标：把规则整理成大模型可读的输入。

你可以切换 Prompt 模板（A/B/C），但要记住：
- 做对照时只改 Prompt，其他参数保持一致。

---

### 第 5 步：在线查询拿到候选策略
目标：得到 10 份可解析的策略输出（同口径对照）。

关键点：
- 支持限流、重试、分片、输出规范化。
- 输出会被整理成可解析树结构，便于后续训练。

---

### 第 6 步：规则落地
目标：把在线答案转成训练阶段能直接调用的规则。

做到这一步，说明“LLM 输出 -> 可训练规则”链路打通了。

---

### 第 7 步：误差修正训练
目标：训练每份候选策略对应的模型输出。

这一步是 DeLTa 的核心思想之一：
- 不是直接替代基线，而是做“修正增强”。

---

### 第 8 步：融合评估
目标：和基线融合后，统一看 Accuracy/AUC。

你会得到最终日志，用来回答一个关键问题：
- 这套策略到底比纯 RF 基线好多少？

---

### 第 9 步：做 Prompt A/B/C 对照
目标：固定参数，只比较 Prompt 版本。

当前同口径结论（已完成）：
- Prompt A：0.9198
- Prompt B：0.9154
- Prompt C：0.9153

结论：
- 目前 A 最优，B 和 C 接近。

---

## 已跑通结果（你可以直接复用）

- MNIST 全链路：已跑通。
- Prompt-C 同口径评估：已补齐。
- 当前最佳模板：Prompt-A（`mnist_stable.py`）。
- `MNIST` 无 LLM 叶子扩展原型：已能运行，首轮 `baseline_accuracy=0.4953`，`expanded_accuracy=0.5413`。
- `MNIST` 最小版 `LLM-Guided Leaf Expansion`：已能运行，当前 `mock_llm` 口径下 `baseline_accuracy=0.4953`，`expanded_accuracy=0.5603`。
- `MNIST` 四组协议库存摘要：已生成 `DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`，可统一查看 `Baseline RF / Deeper RF / without_llm / llm_guided` 当前已有产物。
- `MNIST` 四组协议同口径摘要：已生成 `DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`，当前统一 `test_accuracy` 结果为 `baseline_rf=0.6908`、`deeper_rf=0.9025`、`without_llm=0.5400`、`llm_guided(mock_llm)=0.5603`、`online_llm=0.5622`。
- `MNIST` 正式在线 `LLM relation` 入口：已接入 `online_llm`，可通过 `OPENAI_*` 环境变量或 `--llm_*` 参数发起真实 relation 排序请求。
- `MNIST` 真实 `online_llm` 同口径尝试：已生成 `DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json`；引入 richer feature prompt、strict validation 与 top-k 试分裂后，当前最新状态为 `success`，`test_accuracy=0.5622`，相对 `mock_llm=0.5603` 的当前差值为 `+0.0019`。
- `MNIST` online relation 重复实验：已生成 `DeLTa-main/results/mnist/mnist_online_llm_repeatability_summary.json`；当前 5 次重复全部为 `0.5622`，`std=0.0`，5/5 次都高于 `mock_llm=0.5603`。
- `MNIST` oracle 单特征上界分析：已生成 `DeLTa-main/results/mnist/mnist_leaf_expansion_oracle_analysis.json`；当前 `leaf 8` 的 oracle 最优特征是 `347`，真实 `online_llm` 已选中该特征。
- `bank` 第一次真实迁移验证：已跑通“基线 -> Prompt -> 本地规则训练 -> 融合评估”最小闭环，当前单答案口径 `RF Accuracy=0.8986`，`Fused Accuracy=0.8991`。
- `bank` online relation 迁移验证：已跑通 `without_llm / mock_llm / online_llm` 三条 leaf expansion 原型线，并生成 `DeLTa-main/results/bank/bank_leaf_expansion_same_metric_summary.json`；当前 `without_llm=0.8886`、`mock_llm=0.8890`、`online_llm=0.8890`，`online_llm_minus_mock_llm=0.0`。这说明 bank 上的 online 链路不仅接线正确，而且已通过最新 prompt 增强追平当前 `mock_llm`。
- `bank` 更强语义 prompt + 叶子级对比：当前已把 `task_intro`、`target_outcomes`、`feature_description`、bank 域内指引写入 prompt，并新增 `DeLTa-main/tools/compare_bank_leaf_expansion_online_vs_mock.py`。这一步先把主要差异收敛到 `leaf 11`，证明问题集中在局部特征语义选择，而不是全链路接入。
- `bank` target-conditioned + counterfactual prompt 增强：当前又补入 `predicted_outcome`、`alternative_outcomes`、`category_values_preview`、`predicted_class_top_categories`、`one_feature_train_accuracy`，并加入显式 counterfactual 指令去比较类似 `age` 与 `duration` 的取舍。最新叶子级结果显示 `leaf 11` 上 online 已从 `duration` 翻转为 `age`，与 mock 对齐，从而把全局 `online_llm` 提升到 `0.8889748977109366` 并追平 `mock_llm`。
- `bank` 叶子优化优先级分析：已新增 `DeLTa-main/tools/analyze_bank_leaf_optimization_priority.py` 并生成 `DeLTa-main/results/bank/bank_leaf_expansion_optimization_priority.json`；当前 3 个已选叶子在“单特征局部最优”视角下都没有正向 headroom，这说明继续堆 bank 专用 pairwise prompt 的收益已经很有限，后续若想超过 `mock_llm`，更值得优先检查候选叶子选择或扩展结构本身，而不是继续在同一批叶子上微调措辞。
- `bank` 叶子选择策略重排：已新增 `DeLTa-main/tools/analyze_bank_leaf_selection_strategies.py` 并实跑 `impurity` 策略；当前 `without_llm + impurity = 0.8904`、`mock_llm + impurity = 0.8901`，都高于旧默认策略，说明 `bank` 的下一步优先级已经切到“换叶子”，不是“继续修旧叶子文案”。
- `adult` 接入审计：已新增 `DeLTa-main/tools/audit_dataset_integration_readiness.py` 并生成 `DeLTa-main/results/adult/adult_integration_readiness.json`；当前确认 `adult` 的代码入口已具备，但数据目录、实验台账和模块映射仍缺，因此下一步应先补数据层，而不是继续补说明文档。
- `car` 正式协议：已完成第三个完整样板闭环；当前 `online_llm=0.8257`，已超过 `without_llm=0.8043`、`mock_llm=0.7911`、`Deeper RF=0.7961` 与传统 `RF=0.8174`，但仍低于传统 `DeLTa=0.8470`。
- `jannis` 正式协议：已完成第四个完整样板闭环；当前 `online_llm=0.5814`，已超过 `prototype_baseline=0.5667`、`without_llm=0.5640`、`mock_llm=0.5651`，但仍低于 `Deeper RF=0.6324`、传统 `RF=0.6488` 与传统 `DeLTa=0.6575`。
- `house_16H_reg` 正式协议：已完成第 5 个样板（第 1 个回归样板）主对比；当前 `online_llm=42044.3802`，已优于 `baseline=43327.9766`、`without_llm=42303.1098` 与 `mock_llm=42843.3905`，但仍低于 `Deeper RF=38943.8822`、`traditional_rf=35230.4577` 与 `traditional_delta(cart回退)=41434.7702`。
- 后续主对比协议将优先固定为 `Deeper RF (depth=4)`、`depth=3 + without_llm`、`depth=3 + online_llm` 三组；其中 `depth=3` 保留为公共底座锚点，`mock_llm` 作为辅助 sanity check。

---

## 常见问题

1. **在线查询报错 401 / key 问题**
   - 先检查环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。
   - 如果不想每次启动都手动设置，可以直接在 `DeLTa-main/local_openai_config.json` 里维护本地私有配置，代码会自动加载。

2. **在线查询返回不稳定**
   - 增加重试、请求间隔，必要时启用规则分片。

3. **叶子扩展原型的 online_llm 跑不起来**
   - 先检查环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。
   - 也可以检查 `DeLTa-main/local_openai_config.json` 是否已填好本地私有配置。
   - 再检查参数：`--relation_source online_llm`、`--llm_model`、`--llm_max_retries`、`--llm_retry_delay`、`--llm_request_interval`、`--top_k_features_to_try`。
   - 如果 `DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json` 里是 `status = success`，优先看 `test_accuracy`、`mnist_leaf_expansion_same_metric_summary.json` 和 `mnist_leaf_expansion_oracle_analysis.json`；如果是 `status = failed` 且出现 `401` / `invalid_api_key`，优先检查当前 key 是否有效；如果是 `status = blocked` 且出现 `Cloudflare`，再按网络拦截方向排查。

4. **结果看起来“变好了/变差了”但不确定**
   - 一定做同口径对照：相同数据、相同参数、仅改 Prompt。

---

## 接下来怎么继续

如果是第一次接手这个项目，建议按这个顺序：
1. 先看 `docs/lookme.md` 勾选项（知道当前进度）。
2. 再看 `Update.md` 最近两天记录（知道做过什么）。
3. 再看 `docs/stage1_mnist_semigeneral_guide.md`（知道当前阶段该改哪里、先调什么）。
4. 再看 `docs/stage1_bank_migration_guide.md`（知道第一个非图像数据集为什么选 `bank`、实际迁移先做什么）。
5. 再看 `docs/mnist_llm_guided_leaf_expansion_prototype.md`（知道 `MNIST` 上的新方法原型怎么正式开）。
6. 再看 `DeLTa-main/results/mnist/mnist_online_llm_repeatability_summary.json`（知道 `+0.0019` 是否稳定）。
7. 再看 `DeLTa-main/results/mnist/mnist_leaf_expansion_oracle_analysis.json`（知道哪些叶子已经接近单特征上界）。
8. 最后看实验台账（知道哪些结论已经可靠）。

这样可以在最短时间内接上当前项目节奏。