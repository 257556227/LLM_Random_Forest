> 总顺序固定：**DeLTa复现 -> TnT单数据集闭环 -> 源码迁移 -> 兼容跑通 -> Prompt优化**。

## 阶段 1：先把 DeLTa 跑明白（先学，再改）

执行清单（照着做）：

- [x] 跑通这 6 个脚本：`run_randforest.py` -> `run_get_prompt.py` -> `run_get_answer.py` -> `get_trees.py` -> `run.py` -> `run_ensemble.py`
- [x] 画一张模块图：`Rule Extractor` / `LLM Interface` / `Gradient Net Trainer` 各在什么脚本里
- [x] 记录每一步输入和输出文件路径（日志、prompt、answer、rule、result）
- [x] 至少完整跑 1 次并保存日志（后面迁移要对照）

---

## 阶段 2：在 1 个 TnT 数据集上打通闭环

执行清单（照着做）：

- [x] 先选 1 个 TnT 数据集（不要多选）
- [x] 跑通整条链：训练规则 -> 组 Prompt -> LLM 输出 -> 规则落地 -> 训练评估
- [x] 每一步都留产物（日志、规则文件、结果文件）
- [x] 先只追求"跑通"，不追求最优分

---

## 阶段 3：迁移 DeLTa 可复用源码

执行清单（照着做）：

- [x] 一次只迁 1 个模块（不要一起改）
- [x] 优先迁这 3 块：规则抽取、Prompt 构造与解析、误差修正训练链路
- [x] 每迁 1 次就跑最小样例验证
- [x] 每次迁移都保留回退点（可快速撤回）

---

## 阶段 4：做兼容并完整跑通

执行清单（照着做）：

- [x] 检查数据兼容：分类/回归、数值/类别特征
- [x] 检查规则兼容：LLM 输出能解析、能执行
- [x] 检查训练兼容：超参、随机种子、日志路径统一
- [x] 连续跑多次，确认不崩；输出有 RF 基线 + 迁移后结果

---

## 阶段 5：最后再做 Prompt 优化

执行清单（照着做）：

- [x] 参考 Prompt1.png和Prompt2.png，形成 Prompt-C（Prompt1/2 对齐版）同口径评估并补齐对照结论
- [x] 固定其它变量，只改 Prompt
- [x] 每版 Prompt 记录同一套指标
- [x] 每个配置跑 10 次，取均值再比较

---

## 最终验收（只看这 5条）

- [x] 单数据集全链路稳定复现
- [x] 迁移后结果可与 RF 基线对比
- [x] Prompt 优化有量化收益（不是体感）
- [x] 全部过程有日志和产物可回查
- [x] 参考流程图.png形成一个我们模型的整体流程图

---

## 当前追加阶段：MNIST 半通用化（当前重点）

执行清单（照着做）：

- [x] 固定 `MNIST` 为阶段一样板，只维护一套标准入口，不同时扩多个数据集
- [x] 把"改数据去哪里改、调参数先看哪里"固化到 `docs/stage1_mnist_semigeneral_guide.md`
- [x] 明确数据入口、Prompt入口、规则入口、训练入口、实验记录入口分别对应哪些文件
- [x] 保持 `docs/readme.md`、`docs/lookme.md`、`Update.md` 与操作文档四者同步
- [x] 已确定第 1 个非图像迁移对象为 `bank`，并新增迁移执行说明 `docs/stage1_bank_migration_guide.md`
- [x] 已正式开出 `MNIST` 上的 `LLM-Guided Leaf Expansion Forest` 原型说明 `docs/mnist_llm_guided_leaf_expansion_prototype.md`
- [x] 已按原型说明完成 `MNIST` 上的第一版 `Leaf Expansion without LLM` 对照实验（首轮 `0.4953 -> 0.5413`）
- [x] 已在当前原型入口上接入最小版 `LLM relation` 并形成第一版 `LLM-Guided Leaf Expansion` 对照实验（当前 `mock_llm` 口径 `0.4953 -> 0.5603`）
- [x] 已补齐阶段一速查表，并完成可回放 `json relation` 版本的 `LLM-Guided Leaf Expansion` 运行

---

## 2026-03-24: 8数据集全面测试完成

### 最终成果 (5/8 超越 deeper_rf)

| 数据集 | 类型 | LLM最佳 | deeper_rf | 状态 |
|--------|------|---------|-----------|------|
| bank | 二分类 | 89.04% | 88.73% | ✅ |
| adult | 二分类 | 84.93% | 84.56% | ✅ |
| car | 多分类 | 82.57% | 79.61% | ✅ |
| house_16H_reg | 回归 | 41923 | 42440 | ✅ |
| california_housing | 回归 | 0.6468 | 0.6508 | ✅ |
| credit-g | 二分类 | 72.86% | 73.14% | ❌ 差0.28% |
| jannis | 多分类 | 61.56% | 63.24% | ❌ |
| mnist | 多分类 | 62.02% | 81.64% | ❌ |

### 执行清单

- [x] 8个数据集 formal protocol 全部完成
- [x] 网格调参覆盖 50+ 种配置
- [x] 5/8 数据集超越 deeper_rf
- [x] 更新 EXPERIMENT_SUMMARY.md
- [x] 更新实验台账 (credit_g, jannis)
- [x] 推送 PR 分支

### 下一步

- [ ] 使用真实 LLM 测试 credit-g (差0.28%即可突破)
- [ ] 改进 jannis 方法 (高维匿名特征)
- [ ] 整理 readme.md 和 lookme.md
- [x] 已接入正式在线 `LLM relation` 入口（`online_llm`），支持通过 `OPENAI_*` 与 `--llm_*` 参数控制真实 relation 请求
- [x] 已补四组协议统一库存摘要（`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`），并明确当前还不是严格同口径最终结论
- [x] 已用统一 runner 真实执行四组库存协议的全部阶段（`baseline_rf / deeper_rf / without_llm / llm_guided / build_summary`）
- [x] 已生成四组协议统一 `test_accuracy` 同口径摘要（`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`）
- [x] 已完成真实 `online_llm` 同口径正式尝试并落盘状态记录（`DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json`，当前最新为 `success`，`test_accuracy=0.5622`）
- [x] 已按 `bank` 迁移说明完成第一轮真实链路验证并留存产物（当前采用本地规则文件替代在线答案，单答案融合 `0.8986 -> 0.8991`）
- [x] 已在可用网络/凭据条件下跑通 `online_llm` 并形成非 `mock/json` 口径的真实对照实验
- [x] 已完成 richer prompt / strict validation / top-k 试分裂 / oracle 单特征上界分析，并确认当前 `online_llm=0.5622` 已略高于 `mock_llm=0.5603`
- [x] 已完成 `MNIST` online relation 重复实验增强，并确认 `+0.0019` 在当前 5 次重复下稳定（5/5 次为 `0.5622`）
- [x] 已把当前 online 版 relation 迁到 `bank`，并生成 `bank_leaf_expansion_same_metric_summary.json`
- [x] 已完成一轮轻量 `bank` 专用 prompt 增强：去掉伪 `pixel` 语义，并补入 tabular business data / numeric / categorical 指引
- [x] 已完成一轮更强 `bank` 语义 prompt 增强：加入 `task_intro`、`target_outcomes`、`feature_description` 与 bank 领域指引
- [x] 已完成一轮 target-conditioned `bank` prompt 增强：加入 `predicted_outcome`、`alternative_outcomes`、类别值预览与类条件 top categories
- [x] 已新增 `online-vs-mock` 叶子级对比工具：`DeLTa-main/tools/compare_bank_leaf_expansion_online_vs_mock.py`
- [x] 已基于叶子级对比继续分析 `leaf 11`，并补入 `one_feature_train_accuracy` + explicit counterfactual prompt；当前 online 已从 `duration` 翻转到 `age`，`bank online_llm=0.8889748977109366` 追平 `mock_llm`
- [x] 已完成 `bank` 叶子优化优先级筛查，并生成 `DeLTa-main/results/bank/bank_leaf_expansion_optimization_priority.json`
- [x] 已完成 `bank` 候选叶子选择策略预览，并确认 `impurity` 会切到新叶子 `14 / 10 / 6`
- [x] 已完成 `bank` 的最小 `impurity` 实跑：当前 `without_llm=0.8904`、`mock_llm=0.8901`，均高于旧默认策略
- [x] 已完成 `bank` 的 `online_llm + impurity`：当前 `online_llm=0.8901`，与 `mock_llm` 追平，但仍未超过 `without_llm`
- [ ] 固定后续主对比协议为 `Deeper RF / without_llm / online_llm`，把 `depth=3` 作为底座锚点、`mock_llm` 作为次优先级 sanity check
- [ ] 从现在开始主线切到"8 个公开数据集完整闭环优先"：先把每个数据集都推进到 readiness + 三组原型 + 正式协议 + 台账 + 测试齐全，再统一回看哪些已经超过 `deeper_rf`
- [ ] 在 8 数据集扩展阶段，只保留高收益动作：补数据层/readiness、换候选叶子、补局部质量信号、扩大局部结构搜索；低收益微调先不做
- [x] 已完成 `adult` 接入就绪审计，并生成 `DeLTa-main/results/adult/adult_integration_readiness.json`
- [x] 已补齐 `adult` 数据目录与 `info.json`，并真实生成 `DeLTa-main/example_datasets/adult/`
- [x] 已完成 `adult` 第一轮强语义表格原型对照：`without_llm=0.8468`、`mock_llm=0.8441`、`online_llm=0.8493`
- [x] 已完成 `adult` 的正式主对比协议摘要：`deeper_rf=0.8456`、`traditional_rf=0.8585`、`traditional_delta=0.8602`，当前 `online_llm=0.8493` 已超过 `Deeper RF`，但仍低于传统主链
- [x] 已完成 `credit-g` 数据目录与 `info.json`，并真实生成 `DeLTa-main/example_datasets/credit-g/`
- [x] 已完成 `credit-g` 第一轮原型对照：`prototype_baseline=0.7286`、`without_llm=0.7171`、`mock_llm=0.7286`、`online_llm=0.7171`
- [x] 已完成 `credit-g` 正式主对比协议：`deeper_rf=0.7314`、`traditional_rf=0.7429`、`traditional_delta=0.7343`，当前 `online_llm=0.7171` 仍低于全部主对照
- [x] 已完成 `car` 第三数据集完整闭环：`online_llm=0.8257`、`deeper_rf=0.7961`、`traditional_rf=0.8174`、`traditional_delta=0.8470`
- [x] 已完成 `car` 的小范围调优收口：默认 `impurity_mass + top_k_leaves=3 + top_k_features_to_try=3` 当前最优，`sample_count` 更差，`impurity` 无额外收益
- [x] 已沉淀 `docs/dataset_generalization_methodology.md`，用于后续迁 `jannis / 回归`
- [x] 已完成 `jannis` 第四数据集完整闭环：`online_llm=0.5814`、`deeper_rf=0.6324`、`traditional_rf=0.6488`、`traditional_delta=0.6575`
- [x] 已完成 `jannis` 的小范围调优收口：默认 `impurity_mass + top_k_leaves=3 + top_k_features_to_try=3` 当前最优，`impurity` 更差，`sample_count / top4 / topk4` 仅追平默认
- [x] 已完成 `house_16H_reg` 第 5 个样板（第 1 个回归样板）首轮闭环：三组原型、同口径摘要、台账、模块映射、集成测试已齐全
- [x] 已完成 `house_16H_reg` 正式协议收口：`online_llm=42044.3802`，但仍低于 `deeper_rf=38943.8822`、`traditional_rf=35230.4577` 与 `traditional_delta(cart回退)=41434.7702`
- [x] 已把 `house_16H_reg` 记为"回归原型有效但正式主对比未翻盘"的正式反例
- [ ] 下一步优先切 `california_housing`，形成第二个回归样板，并验证当前回归正式反例是否具有普遍性
- [ ] 对已完成样板统一记录三类结论：已经超过 `deeper_rf`、只超过原型但没超过传统对照、当前正式反例

### 当前文档收口动作

- [x] 已确认 `docs/lookme_origin.md`、`docs/lookme整体步骤.md`、`docs/draft.md` 属于历史过程文档
- [x] 已将上述 3 份历史文档低风险归档到 `docs/archive/`，主入口只保留索引说明

---

## 执行总结（一步一步照着做）

### 当前追加规则（2026-03-11 起执行）

1. 先扩样板，再追极限分数；
2. 每个数据集先补到"可维护状态"，再决定是否继续调优；
3. 只优先做高收益动作，不在单个数据集上长期做低收益细磨；
4. 如果高收益动作做完仍明显低于传统主对照，就先记正式反例，后面统一再讨论是否继续追。

1. 固定单一数据集与评估口径，只保留一套标准指标。
2. 完成特征抽取与预处理，确保训练/验证/测试划分稳定。
3. 训练基线分类器，得到首版可解释规则。
4. 将规则转为 Prompt 输入，约束输出格式为可解析树结构。
5. 调用 LLM 生成候选策略（先追求可解析、可执行，再追求高分）。
6. 做规则映射与落地，保证每条策略可被训练阶段直接调用。
7. 执行误差修正训练，产出各候选策略对应预测结果。
8. 与基线进行融合评估，统一对比准确率/稳定性等核心指标。
9. 固定其余变量，仅做 Prompt A/B/C 版本对照，按同口径复现实验。
10. 记录结论并固化复现清单，形成可迁移、可回查、可排期的闭环。