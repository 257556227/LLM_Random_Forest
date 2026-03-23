# MNIST：LLM-Guided Leaf Expansion Forest 原型说明

## 1. 这份文档的定位

这份文档回答的问题只有一个：

> 当阶段一底座收口到“够稳定”之后，`MNIST` 上的第一版 `LLM-Guided Leaf Expansion Forest` 原型应该怎么正式开工。

它不是最终论文文档，也不是结果汇报文档，而是“第一版方法原型”的执行说明。

---

## 2. 原型目标

这一版原型不追求一步到位做成完整新框架，而是先验证一个核心问题：

> 让 LLM 参与叶子末端的局部扩展，是否有机会在不大改 RF 主干的前提下，提升单棵 base learner 的局部表达能力。

因此，这一版只做最小可验证机制：
- 保留原始 RF 前部主干；
- 只在候选叶子上追加 1 层局部扩展；
- 由 LLM 提供候选特征排序或候选关系；
- 阈值仍由本地数据上的 stump / CART 计算；
- 先在 `MNIST` 上验证，再决定是否迁到 `bank`。

---

## 3. 这一版不做什么

为了避免第一版原型失控，这一版明确不做：
- 不直接重写整个随机森林训练器；
- 不一上来做 tree-to-graph；
- 不让 LLM 直接生成完整整棵树；
- 不让 LLM 直接拍阈值；
- 不同时引入多种 relation 来源和多套复杂搜索策略。

一句话：

> 第一版只证明“LLM 引导的局部叶子扩展”有没有信号，不证明最终系统已经完成。

---

## 4. 原型方法边界

### 4.1 输入
- 当前 `MNIST` 的已训练 RF 或单棵树；
- 叶子节点对应的样本子集；
- 来自规则、特征统计或样本描述的 LLM 输入材料。

### 4.2 输出
- 每个候选叶子的候选扩展特征排序；
- 基于本地 stump 求得的局部 split；
- 更新后的扩展树或扩展规则表示；
- 和 baseline / deeper RF 对照的评估结果。

### 4.3 最小闭环
这一版至少要形成下面闭环：
1. 选叶子；
2. 组织叶子上下文；
3. 调 LLM 或模拟 relation 输出；
4. 本地算局部 split；
5. 写回树结构；
6. 跑评估并记录结果。

---

## 5. 代码落点建议

当前建议先不大改已有主链，而是在现有代码旁边开一个尽量独立的原型层。

### 5.1 建议优先复用的现有文件
- `DeLTa-main/model/randomforest_save.py`：提供当前 RF 生长基线；
- `DeLTa-main/model/utils.py`：已有叶子索引、子树替换、叶子分配等工具基础；
- `DeLTa-main/run_randforest.py`：当前基线入口；
- `DeLTa-main/results/mnist/`：当前 `MNIST` 基线结果产物位置。

### 5.2 建议新增的原型层文件
- `DeLTa-main/model/leaf_expansion/`：放原型逻辑；
- `DeLTa-main/model/leaf_expansion/mnist_leaf_selector.py`：候选叶子筛选；
- `DeLTa-main/model/leaf_expansion/mnist_relation_provider.py`：LLM relation 或模拟 relation 提供；
- `DeLTa-main/model/leaf_expansion/mnist_leaf_expander.py`：本地 split 计算与写回；
- `DeLTa-main/run_leaf_expansion_mnist.py`：原型实验入口。

第一版不一定一次把这些文件全写完，但建议按这个职责切开，不要把所有逻辑塞回单个旧文件里。

---

## 6. 顶级程序员视角的原型拆解

### 任务 A：先做无 LLM 的叶子扩展 baseline
- [x] 从当前 `MNIST` 基线树中筛出候选叶子；
- [x] 对候选叶子额外补 1 层普通 impurity split；
- [x] 形成第一版 `Leaf Expansion without LLM` 对照结果。

验收标准：
- 能回答“单纯多扩一层”到底有没有收益。

当前首轮结果（2026-03-07）：
- baseline accuracy = `0.4953`
- expanded accuracy = `0.5413`
- 结果文件：`DeLTa-main/results/mnist/leaf_expansion_without_llm_summary.json`

### 任务 B：定义 LLM 需要看到什么上下文
- [x] 已固定第一版叶子级输入格式：叶子 id、样本数、impurity、类别分布、候选特征摘要；
- [ ] 决定是直接发原始规则，还是发压缩后的叶子摘要；
- [ ] 先固定一版 prompt，不在这一阶段做 Prompt 大搜索。

验收标准：
- 对同一叶子，LLM 输入结构稳定且可重复生成。

当前已落地的最小接口：
- 叶子 prompt 构造函数：`build_llm_leaf_prompt`
- relation 来源：`mock_llm` / `json` / `online_llm`
- 当前已验证：`mock_llm` / `json`
- 当前已接入且已经完成一轮正式尝试：`online_llm`

当前 `online_llm` 入口约定：
- 通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 控制在线模型；
- 本地自用场景也支持通过 `DeLTa-main/local_openai_config.json` 自动加载这三项配置，不必每次启动手动设置环境变量；
- 也可在原型入口用 `--llm_model`、`--llm_max_retries`、`--llm_retry_delay`、`--llm_request_interval` 单独覆盖；
- 摘要结果会额外记录 `llm_model` 与 `llm_raw_responses`，便于回放和排查。

### 任务 C：实现 relation 到 split 的转换
- [x] 已让最小版 relation 接口只输出候选特征排序，不直接输出阈值；
- [x] 已在本地对候选特征跑 stump / CART 求阈值；
- [x] 已把局部 split 写回扩展树结构。

验收标准：
- 原型能清楚区分“LLM 负责方向”与“本地数据负责阈值”。

当前首轮结果（2026-03-07，`mock_llm`）：
- baseline accuracy = `0.4953`
- llm_guided expanded accuracy = `0.5603`
- 结果文件：`DeLTa-main/results/mnist/llm_guided_leaf_expansion_summary.json`

当前已完成一轮可回放 `json relation` 验证（2026-03-07）：
- relation_source = `json`
- llm_guided expanded accuracy = `0.5603`
- relation 文件：`DeLTa-main/configs/leaf_expansion/mnist_mock_relation.json`
- 结果文件：`DeLTa-main/results/mnist/llm_guided_leaf_expansion_json_summary.json`

当前也已补上正式在线 relation 接口（2026-03-07）：
- relation_source = `online_llm`
- 入口：`python run_leaf_expansion_mnist.py --mode llm_guided --relation_source online_llm --llm_model <model> --save_summary`
- 说明：本轮代码已接好正式在线 LLM provider，也就是“正式在线 LLM relation”入口；但是否能产出真实对照结果仍取决于当前环境是否提供可用的 `OPENAI_*` 凭据与网络。

### 任务 D：固定第一版对照实验
- [x] 已把现有 `Baseline RF` 日志纳入统一协议摘要；
- [x] 已把现有 `Deeper RF` 日志纳入统一协议摘要；
- [x] 跑 `Leaf Expansion without LLM`；
- [x] 跑第一版 `LLM-Guided Leaf Expansion`（当前为 `mock_llm` relation 接口验证）。
- [x] 已生成严格同口径的 `test_accuracy` 摘要：`Baseline RF / Deeper RF / without_llm / llm_guided`。
- [x] 已把真实 `online_llm` 的正式尝试状态纳入同口径摘要。
- [x] 已拿到 richer prompt + strict validation + top-k 试分裂后的真实 `online_llm` 成功精度结果：`test_accuracy = 0.5622`。
- [x] 已补 oracle 单特征上界分析，并确认当前 `online_llm` 已略高于 `mock_llm = 0.5603`。
- [ ] 还需要继续判断当前 `+0.0019` 是否稳定，还是只是一轮在线波动。

验收标准：
- 至少能回答：收益来自“树更深”，还是来自“LLM 引导的更好扩展”。

当前推荐的主对比协议已经收敛为：
- `Deeper RF (depth=4)`
- `depth=3 + without_llm`
- `depth=3 + online_llm`

补充说明：
- `depth=3` 继续保留，但更适合作为公共底座锚点；
- `mock_llm` 继续保留，但后续优先级下降，更适合作为 sanity check；
- `without_llm` 和 `Deeper RF` 不是一个意思：前者是局部扩展，后者是全树统一加深。

当前第一版统一协议摘要（2026-03-08）：
- 汇总文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`
- 汇总脚本：`DeLTa-main/tools/build_mnist_protocol_summary.py`
- 统一 runner：`DeLTa-main/tools/run_mnist_four_way_protocol.py --dry_run`
- 协议配置：`DeLTa-main/configs/leaf_expansion/mnist_four_way_protocol.json`
- 当前状态：`strict_same_metric_protocol_ready = false`
- 解释：四组结果已经能统一盘点，但 `Baseline RF / Deeper RF` 当前来自 DeLTa 主链 fused eval，`without_llm / llm_guided` 当前来自叶子扩展原型 single-model eval，因此这份摘要是“阶段一库存版协议”，不是最终 apples-to-apples 结论。

当前同口径摘要（2026-03-08）：
- 文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`
- 脚本：`DeLTa-main/tools/evaluate_mnist_same_metric_protocol.py`
- 口径：`metric_scope = same_test_accuracy_eval`
- 当前状态：`strict_same_metric_protocol_ready = true`
- 当前结果：
	- `baseline_rf = 0.6908`
	- `deeper_rf = 0.9025`
	- `without_llm = 0.5400`
	- `llm_guided(mock_llm) = 0.5603`
	- `online_llm = 0.5622`

当前 online 同口径尝试产物：
- 文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json`
- 脚本：`DeLTa-main/tools/attempt_mnist_online_llm_same_metric.py`
- 作用：无论真实在线调用成功还是被凭据/网络阻塞，都会把状态正式落盘，而不是只停留在口头说明。
- 当前状态（2026-03-08）：`status = success`
- 当前结果：`baseline_accuracy = 0.4953`，`test_accuracy = 0.5622`，`llm_model = gpt-4o`。
- 当前补充说明：同口径摘要已新增 `experiments.online_llm`，并记录 `online_llm_minus_mock_llm = +0.0019`，说明真实在线 relation 已正式进入统一对照，并首次略高于 `mock_llm`。
- 当前方法增量：prompt 已补入候选特征的 `pixel`、`information_gain`、`class_mean_spread`、`predicted_class_mean_gap`；同时在线返回若出现候选外 id 或重复 id 会直接重试；扩展器不再只吃 top-1，而是会在 top-k 候选里试分裂再择优。
- 历史排查说明：这条链路早先确实经历过 `Cloudflare` 拦截与官方端点下的 `401 invalid_api_key`，但这些问题已经不是当前最新状态。

当前 oracle 对照产物：
- 文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_oracle_analysis.json`
- 脚本：`DeLTa-main/tools/analyze_mnist_leaf_oracle.py`
- 当前结论：
	- `leaf 8` 的 oracle 最优单特征是 `347`；
	- `online_llm` 当前已经选中 `347`，`oracle_gap = 0.0`；
	- `mock_llm` 在 `leaf 8` 选中 `374`，离 oracle 还差 `0.0131`；
	- `leaf 9` 的多个候选特征局部效果并列，`without/mock/online` 在该叶子上几乎打平。

### 任务 E：记录失败也能复用
- [x] 已把真实 relation 的正式尝试产物落到 `mnist_leaf_expansion_online_llm_attempt.json`，并支持成功/失败/阻塞统一落盘；
- [x] 已把负结果写回 `Update.md` 和实验台账；
- [x] 已证明后续迁移到 `bank` 时可以先做轻量 prompt 适配：去掉伪 `pixel` 语义并补 tabular 指引，而不需要重写整条链；
- [ ] 还需要判断为什么这类轻量适配在 `bank` 上语义更合理，但精度仍未超过 `mock_llm`。

验收标准：
- 即使结果不升，也能明确知道卡在“选叶子 / relation / 本地 split / 写回 / 评估”的哪一层。

---

## 7. 最小执行顺序

建议按这个顺序正式开工：

1. 先补无 LLM 的 `Leaf Expansion without LLM` 基线；
2. 再定义叶子级输入摘要格式；
3. 再接一版最小化的 LLM relation 输出；
4. 用本地 stump 算阈值并写回；
5. 跑四组统一对照；
6. 把结果写回日志与实验记录。

不要一开始就：
- 同时改叶子筛选；
- 同时改 Prompt；
- 同时改 relation 形式；
- 同时改评估协议。

否则结果很难解释。

---

## 8. 第一版完成标准

满足下面 6 条，才算 `MNIST` 原型真正开起来：
- 有单独的原型入口，不和旧链路混在一起；
- 有 `Leaf Expansion without LLM` 对照；
- 有 `LLM-Guided Leaf Expansion` 对照；
- 有清晰的叶子级输入和输出格式；
- 有统一评估记录；
- 有负结果也能回查的日志。

---

## 9. 当前首轮运行结果

目前已经完成一轮真正的无 LLM 原型运行，入口为：
- `DeLTa-main/run_leaf_expansion_mnist.py`

当前首轮结果：
- `prototype = Leaf Expansion without LLM`
- `baseline_accuracy = 0.4953`
- `expanded_accuracy = 0.5413`
- `selected_leaf_ids = [10, 8, 9]`
- `expanded_leaf_count = 3`

这组结果的意义不是“已经达到最终方法水平”，而是：
- 原型入口已经能真正运行；
- 候选叶子筛选 -> 本地特征排序 -> 局部扩展 -> 评估 这条链已经打通；
- 下一步可以在同一入口上继续接入最小版 LLM relation，而不是从零重写。

当前还额外完成了一轮最小版 `LLM-Guided Leaf Expansion`：
- `relation_source = mock_llm`
- `baseline_accuracy = 0.4953`
- `expanded_accuracy = 0.5603`
- 结果文件：`DeLTa-main/results/mnist/llm_guided_leaf_expansion_summary.json`

同时也已完成一轮可回放的 `json relation` 运行：
- `relation_source = json`
- `baseline_accuracy = 0.4953`
- `expanded_accuracy = 0.5603`
- 结果文件：`DeLTa-main/results/mnist/llm_guided_leaf_expansion_json_summary.json`

这里要特别注意口径：
- 这说明“LLM relation 接口位”已经打通；
- 当前 `mock_llm` 和 `json` 都还不是正式在线大模型结论；
- 当前 `online_llm` 已接入正式入口，并且已经拿到一份真实在线尝试成功结果：`test_accuracy = 0.5622`；
- 当前已经不再是“代码没跑到”或“在线阻塞未通”，而是进入“如何把微弱增益放大并验证稳定性”的阶段；
- 当前还新增了一份四组统一库存摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`；
- 但这份库存摘要明确标注了 `strict_same_metric_protocol_ready = false`；
- 当前也已经补出一份同口径摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`；
- 当前同口径摘要里既保留 `online_llm_attempt`，也新增了 `experiments.online_llm`，用于统一管理真实在线结果；
- 当前还新增了一份 `mnist_leaf_expansion_oracle_analysis.json`，用于判断 online/mock 到叶子级单特征上界还有多远；
- 因此它们证明的是“原型结构已经可插拔、可回放”，还不是最终论文结果。

---

## 10. 和阶段一底座的关系

这份文档不是替代 [docs/stage1_mnist_semigeneral_guide.md](docs/stage1_mnist_semigeneral_guide.md)，而是接在它后面。

更准确地说：
- [docs/stage1_mnist_semigeneral_guide.md](docs/stage1_mnist_semigeneral_guide.md) 负责回答“底座是否稳定”；
- [docs/stage1_bank_migration_guide.md](docs/stage1_bank_migration_guide.md) 负责回答“底座是否可迁移”；
- 本文档负责回答“在稳定底座上，`MNIST` 的第一版新方法原型怎么开”。

所以顺序不是互相替代，而是前后衔接。

---

## 11. 当前下一步

现在最推荐的直接动作是：
- 把 `stage1_mnist_semigeneral_guide.md` 视为已完成底座样板；
- 把当前重心正式切到 `stage1_bank_migration_guide.md`，继续把 `bank` 从“追平 mock”推进到“尽量超过 mock 与传统对照”；
- 再以当前 `MNIST` 原型为母版，逐步扩展到更多公开数据集；
- 后续统一朝“8 个数据集、尽量做到 `online_llm > mock_llm` 且 `online_llm > 传统对照`”推进；
- 并把通用化过程中的经验与失败模式持续写入 `docs/generalization_scaling_lessons.md`。

也就是说：

> 不是先追求“像论文一样完整”，而是先让第一版原型可运行、可解释、可失败。