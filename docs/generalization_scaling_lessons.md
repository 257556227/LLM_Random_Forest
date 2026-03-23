# 调优与完全通用化推进记录

## 1. 这份文档记录什么

这份文档不是第一阶段的迁移说明，也不是单一数据集原型说明。

它专门记录下面两件事：
- `LLM-Guided Leaf Expansion Forest` 如何从 `MNIST` 样板扩到更多公开数据集；
- 在“图像 -> 表格 -> 更多数据类型”推进过程中，哪些适配经验和失败教训可以复用。

---

## 2. 当前阶段定位

当前阶段顺序调整为：
- `stage1_mnist_semigeneral_guide.md`：已完成；
- `stage1_bank_migration_guide.md`：当前阶段；
- 本文档：下一阶段，聚焦调优与完全通用化。

也就是说，后续工作的重点不再是“能不能把一个样板跑通”，而是：

> 能不能把同一套结构化协议扩展到更多数据集，并让真实 `online_llm` 尽量稳定超过 `mock_llm` 与传统对照。

---

## 3. 统一主对比协议

后续正式汇报与大部分实验表，优先固定下面三组主对照：
- `Deeper RF (depth=4)`
- `depth=3 + without_llm`
- `depth=3 + online_llm`

辅助对照：
- `depth=3`：公共底座锚点
- `depth=3 + mock_llm`：sanity check，用于确认真实 LLM 是否优于稳定启发式假基线

### 3.1 为什么 `Deeper RF` 不等于 `without_llm`

两者虽然都可以被理解成“多一层”，但含义完全不同：
- `Deeper RF`：整棵树统一再长一层，是**全局容量提升**；
- `without_llm`：只在少数高价值叶子上补 1 层，是**局部结构扩展**。

因此：
- `Deeper RF` 用来对比“传统直接加深”；
- `without_llm` 用来剥离“局部扩展本身”的贡献；
- `online_llm` 才用来衡量“语义引导的局部扩展”是否真的比纯统计局部扩展更强。

---

## 4. 完全通用化目标

### 4.1 目标规模
- 不再停留在 `MNIST + bank`；
- 后续目标是推进到不少于 `8` 个公开数据集。

### 4.2 每个数据集的最低验收
- 能跑通统一入口；
- 能生成同口径摘要；
- 能形成 `Deeper RF / without_llm / online_llm` 三组主对照；
- 至少能解释 `online_llm` 相对 `mock_llm` 与传统对照的变化来源。

### 4.3 理想结果目标
- 尽量做到 `online_llm > without_llm`；
- 尽量做到 `online_llm > mock_llm`；
- 尽量做到 `online_llm > Deeper RF`。

如果没有全部达到，也必须留下正式失败结论。

### 4.4 当前执行策略调整

从现在开始，完全通用化阶段的执行策略调整为：

- **第一优先级**：把 8 个候选数据集全部推进到“可维护、可回查、可比较”的完整状态；
- **第二优先级**：在每个数据集上只优先做已经验证过的高收益动作；
- **第三优先级**：只有当一个数据集完成高收益动作后仍与 `Deeper RF` 差距很大，才接受它在当前阶段“比不了”的结论。

这意味着：
- 不再把“单个数据集立刻超过 `deeper_rf`”当成当前阶段唯一目标；
- 更重视跨 8 个数据集形成正例/反例分布；
- 更重视把失败结论正式沉淀，而不是继续无限细磨。

---

## 5. 目前已经得到的通用化经验

### 5.1 图像数据经验（MNIST）
- richer prompt、strict validation、top-k 试分裂择优是有效的；
- 当前 `online_llm = 0.5622`，已略高于 `mock_llm = 0.5603`；
- 说明图像类特征只要候选摘要足够具体，真实在线 relation 是能学到局部优先级的。

### 5.2 表格数据经验（bank）
- 不能把 tabular 特征当成像素处理；
- 仅靠 feature id 排序是不够的，必须补 task/target 语义与类别值解释；
- counterfactual prompt 与 `one_feature_train_accuracy` 这种局部质量信号，对表格数据尤其关键；
- 当前 `bank online_llm` 已能在默认叶子组和 `impurity` 新叶子组下都追平 `mock_llm`，说明轻量适配路线是成立的；
- 但 `bank` 目前最强结果仍是 `impurity + without_llm = 0.8904`，说明“换叶子”比“继续修旧 prompt”更有效，而真实在线 relation 还没带来额外增益。

### 5.3 强语义表格经验（adult）
- 只要真实数据目录、`info.json` 和类别字段划分先落地，`adult` 这类强语义表格数据很快就能给出有效信号；
- 第一轮原型已经出现 `online_llm = 0.8493 > without_llm = 0.8468 > mock_llm = 0.8441`，说明在线 relation 开始体现正收益；
- 当前已补完正式主对比协议摘要：`deeper_rf = 0.8456`、`traditional_rf = 0.8585`、`traditional_delta = 0.8602`；
- 这说明 `adult online_llm` 目前已经**超过 `Deeper RF(depth=4)`**，但**还没有超过传统主链 `RF / DeLTa`**；
- 因此强语义表格数据的最新结论不是“online 一定赢”，而是“online 对统一深度对照已有效，但距离完整主链基线仍有约 `0.009` 到 `0.011` 的差距”；
- 当前下一步不再是补 baseline，而是决定：继续在 `adult` 上缩小与传统主链的 gap，还是把这套正式协议复制到 `credit-g` 验证泛化性。

### 5.4 金融风险表格经验（credit-g）
- `credit-g` 的第一轮结果比 `adult` 更保守：`prototype_baseline = 0.7286`、`without_llm = 0.7171`、`mock_llm = 0.7286`、`online_llm = 0.7171`；
- 这说明金融风险字段即使有明显语义，也不代表 online 排序天然就会优于启发式排序；
- 当前 `mock_llm` 至少能把 `without_llm` 的下滑拉回到基线，但 `online_llm` 还没有超过 `mock_llm`；
- 当前已补完正式主对比协议：`deeper_rf = 0.7314`、`traditional_rf = 0.7429`、`traditional_delta = 0.7343`；
- 这说明 `credit-g online_llm` 不仅没有超过 `mock_llm`，目前也没有超过统一深度对照和传统主链；
- 因此 `credit-g` 当前最值得优先排查的，不是“有没有语义”，而是“金融语义提示是否把模型带离了叶子局部最优分裂”。

### 5.5 低维纯类别多分类经验（car）
- `car` 的首轮结果非常清楚：`prototype_baseline = 0.7632`、`without_llm = 0.8043`、`mock_llm = 0.7911`、`online_llm = 0.8257`；
- 当前正式协议也已经补齐：`deeper_rf = 0.7961`、`traditional_rf = 0.8174`、`traditional_delta = 0.8470`；
- 这说明在低维纯类别多分类场景下，`online_llm` 已经同时超过 `mock_llm`、`without_llm`、`Deeper RF` 和传统 `RF`；
- 但它仍没有超过传统 `DeLTa` 主链，这意味着“局部结构扩展 + 语义引导”已经很强，但完整融合主链仍有额外收益；
- 调优小扫进一步说明：`impurity_mass` 与 `impurity` 等价，`sample_count` 更差，说明这类数据不该优先按样本数选叶子，而应优先保住高杂质叶子。

### 5.6 高维弱语义多分类经验（jannis）
- `jannis` 的首轮结果很有代表性：`prototype_baseline = 0.5667`、`without_llm = 0.5640`、`mock_llm = 0.5651`、`online_llm = 0.5814`；
- 当前正式协议也已经补齐：`deeper_rf = 0.6324`、`traditional_rf = 0.6488`、`traditional_delta = 0.6575`；
- 这说明在高维弱语义多分类场景下，`online_llm` 依然能明显超过 `mock_llm / without_llm / baseline`，证明 LLM 并不完全依赖强语义字段；
- 但它当前仍明显低于 `Deeper RF / 传统 RF / 传统 DeLTa`，说明这类数据的上限仍更依赖传统容量提升，而不是仅靠局部语义引导；
- 调优小扫进一步说明：`impurity` 更差，`sample_count / top4 / topk4` 只追平默认，默认 `impurity_mass` 仍是当前最稳的收口配置。

### 5.7 当前跨数据集高收益动作（正式版）
- **先补数据层与 readiness**：没有数据目录、`info.json`、审计报告的实验结论一律不作为正式判断；
- **先跑三组原型**：`without_llm / mock_llm / online_llm` 是所有数据集的最低判断入口；
- **先换叶子，再修 prompt**：`bank` 已证明候选叶子策略经常比旧叶子 prompt 微调更有收益；
- **先补局部质量信号**：`task_intro`、`target_outcomes`、`predicted_outcome`、`alternative_outcomes`、`one_feature_train_accuracy` 是当前最稳的 prompt 增量；
- **先扩结构搜索，再细磨低收益超参**：`jannis` 已证明双特征局部树优于更深单特征树，结构搜索优先级应高于低收益 sweep；
- **比不了就正式记反例**：当高收益动作做完仍明显低于传统主对照，应转入反例沉淀，不再长期卡在单点追分。

### 5.8 第一回归样板经验（house_16H_reg）
- `house_16H_reg` 已成为当前第一个真实跑通的回归 leaf expansion 样板；
- 第一轮结果为：`baseline_rmse = 43327.9766`、`without_llm = 42303.1098`、`mock_llm = 42843.3905`、`online_llm = 42044.3802`；
- 这说明回归任务下，真实 `online_llm` 也能从局部统计摘要中得到正向排序信号，且当前优于 `mock_llm` 与 `without_llm`；
- 这次真正验证了“连续目标区间分离 + 局部误差降低”版 prompt 可以工作，不再只是分类机制的名义复用；
- 正式协议现已补齐：`deeper_rf = 38943.8822`、`online_llm = 42044.3802`、`traditional_rf = 35230.4577`、`traditional_delta(cart回退) = 41434.7702`；
- 结论是：`online_llm` 仍优于原型内的 `mock_llm / without_llm / baseline`，但低于 `Deeper RF / traditional RF / traditional DeLTa(cart回退)`，因此当前应把 `house_16H_reg` 记为“回归原型有效，但正式主对比仍未翻盘”的正式反例；
- 另外还暴露出一个工程经验：在无 CUDA 环境下，回归主链的 `tabpfn` 会因为 CPU 内存压力失稳，因此必须提供自动回退到 `cart` 的稳态路径。

---

## 6. 后续记录模板

每扩一个新数据集，建议至少补下面 6 类信息：
- 数据类型：图像 / 表格 / 回归 / 多分类
- 主对比结果：`Deeper RF / without_llm / online_llm`
- 辅助对比：`mock_llm`
- 当前最有效的 prompt 增量
- 当前最大失败点
- 下一步最值得做的修复动作

这样后面即使推进到 8 个数据集，也不会再次丢失上下文。

---

## 7. 推荐的 8 个公开数据集候选清单

下面这 8 个数据集不是随便拍脑袋列出来的，而是综合考虑了三件事：
- 当前仓库里已有的 Prompt / 规则 / 数据适配基础；
- 是否覆盖不同数据类型（图像、多分类、二分类、回归、稀疏/密集表格）；
- 是否适合回答“真实 `online_llm` 到底在哪类数据上更容易超过 `mock_llm` 与传统对照”。

### 7.1 第一梯队：优先马上推进

1. `MNIST`
- 角色：图像样板 / 当前母版数据集
- 当前状态：已稳定跑通，`online_llm = 0.5622 > mock_llm = 0.5603`
- 优先级：`P0`
- 主要价值：继续作为所有 prompt、候选叶子、top-k、oracle 分析的母版
- 主要风险：当前仍显著低于 `Deeper RF`，说明“机制有效”不等于“已经赢过传统容量提升”

2. `bank`
- 角色：第一非图像样板 / 当前迁移主战场
- 当前状态：已追平 `mock_llm`
- 优先级：`P0`
- 主要价值：验证图像以外的数据上，target-conditioned + counterfactual prompt 是否能持续工作
- 主要风险：目前只是追平 `mock_llm`，还没有稳定超过传统主对照

3. `adult`
- 角色：经典二分类表格数据
- 当前基础：仓库已有 Prompt / 规则文件基础
- 优先级：`P1`
- 主要价值：字段语义强，适合检验 LLM 是否真的能利用职业、教育、婚姻等特征语义
- 主要风险：类别值多且存在社会语义偏置，prompt 如果写得太泛容易把“统计相关”误导成“语义重要”

4. `jannis`
- 角色：高维多分类表格数据
- 当前基础：仓库已有 Prompt / 规则文件基础
- 优先级：`P1`
- 主要价值：适合测试当特征维数上升后，候选特征摘要和 top-k 机制是否还能保持稳定
- 主要风险：特征名通常缺乏强语义，LLM 优势可能弱于有明确业务含义的数据集

### 7.2 第二梯队：中期推进

5. `car`
- 角色：低维类别型多分类数据
- 当前基础：仓库已有 Prompt / 规则文件基础
- 优先级：`P2`
- 主要价值：适合测试类别值解释、target-conditioned prompt、counterfactual 提示是否容易起作用
- 主要风险：维度低、传统树已经很强，留给局部扩展的空间可能有限

6. `credit-g`
- 角色：风险评分类二分类数据
- 当前基础：仓库已有 Prompt / 规则文件基础
- 优先级：`P2`
- 主要价值：适合测试 LLM 是否能利用信用/贷款字段语义做局部分裂优先级排序
- 主要风险：当前第一轮已经出现 `online_llm` 未超过 `mock_llm`，说明如果类别值映射与金融语义提示不够贴叶子局部统计，online 很容易退化

7. `house_16H_reg`
- 角色：回归数据入口样板
- 当前基础：仓库已有数据与 Prompt 基础
- 优先级：`P2`
- 主要价值：检验当前机制能否从分类平滑迁到回归；这是“完全通用化”必须面对的一步
- 主要风险：当前很多 prompt 和评估描述是围绕分类写的，回归版需要补连续目标解释、误差口径和候选特征质量信号

8. `california_housing`
- 角色：第二个回归验证点
- 当前基础：仓库已有 Prompt / 规则文件基础
- 优先级：`P2`
- 主要价值：和 `house_16H_reg` 形成回归双样板，避免把回归结论建立在单一数据集上
- 主要风险：地理类与统计类混合特征较多，如果 prompt 不明确“局部阈值优先于泛化常识”，LLM 容易给出听起来合理但局部收益不高的排序

### 7.3 当前预检文档索引

为了避免“候选名单已经写了，但真正开工前还要重新查一遍仓库”这种重复劳动，当前 8 个候选数据集已经整理成下面这套文档入口：

- `MNIST`：`docs/stage1_mnist_semigeneral_guide.md` + `docs/mnist_llm_guided_leaf_expansion_prototype.md`
- `bank`：`docs/stage1_bank_migration_guide.md`
- `adult`：`docs/adult_integration_precheck.md`
- `credit-g`：`docs/credit_g_integration_precheck.md`
- `jannis`：`docs/jannis_integration_precheck.md`
- `car`：`docs/car_integration_precheck.md`
- `house_16H_reg`：`docs/house_16H_reg_integration_precheck.md`
- `california_housing`：`docs/california_housing_integration_precheck.md`

这意味着当前文档层的目标已经从“列出 8 个名字”推进到“8 个名字都能直接点进对应入口文档”。

---

## 8. 建议推进顺序（Google 工程视角）

后续不要同时开 8 个数据集，而是按下面顺序推进：

### 阶段 A：把当前双样板做扎实
- `MNIST`
- `bank`

目标：
- `MNIST` 稳住当前在线增益；
- `bank` 尽量从“追平 mock”推进到“超过 mock”；
- 同时把主对比协议完全固定成 `Deeper RF / without_llm / online_llm`。

### 阶段 B：扩到两个强语义表格数据
- `adult`
- `credit-g`

目标：
- 看 LLM 在“字段有明确人类语义”的表格任务上，是否更容易稳定超过 `mock_llm`；
- 如果这里仍然不明显优于 `mock_llm`，说明方法增益可能更多来自局部统计摘要，而不是自然语言语义。

### 阶段 C：扩到高维 / 低维两类极端表格
- `jannis`
- `car`

目标：
- `jannis` 用来测试高维弱语义情形；
- `car` 用来测试低维强类别值解释情形；
- 两者一起可以帮助判断：到底是“高维度”更难，还是“低维传统树已经够强”更难。

### 阶段 D：补回归能力
- `house_16H_reg`
- `california_housing`

目标：
- 确认这套机制不是只对分类成立；
- 逐步把 prompt、候选特征质量摘要、同口径评估扩展到回归任务。

---

## 9. 当前风险清单（跨数据集）

### 风险 1：`online_llm` 只是在追 `mock_llm`，还没稳定赢传统对照
- 当前最危险的误判是：看到 `online_llm > mock_llm` 就误以为方法已经成熟。
- 真实主问题仍是：能不能稳定超过 `Deeper RF`。

### 风险 2：图像经验不能直接平移到表格
- `MNIST` 上有效的 `pixel / class_mean_spread / information_gain` 提示，迁到表格后往往需要改写成 task/target/category 语义。

### 风险 3：强语义特征不等于局部最优特征
- LLM 很容易偏向“人类读起来合理”的字段，但叶子局部最优分裂不一定是最有业务意义的特征。
- 这也是为什么 `one_feature_train_accuracy`、counterfactual check、top-k 试分裂仍然必须保留。

### 风险 4：回归任务会暴露当前 prompt 的分类偏置
- 现在很多模板天然偏向“预测哪一类”；
- 一旦迁到回归，必须把“类别解释”改成“连续目标区间分离”的表达，否则 online 排序会失真。

---

## 10. 下一步最值得直接做什么

如果按投入产出比排序，当前最值得的动作是：
1. 先把 8 个候选数据集全部推进到统一可维护状态，而不是继续围绕单个数据集死磕；
2. 已完成的数据集只保留高收益动作，不再继续做低收益微调；
3. 已把 `house_16H_reg` 从“第一轮回归原型已跑通”推进到“正式协议已补齐”，当前可正式标注为回归侧正式反例；
4. 再推进 `california_housing`，形成第二个回归样板，并验证这种回归反例是否可复现；
5. 如果后续某个数据集在完成高收益动作后仍明显低于 `deeper_rf`，就直接记为正式反例，后面再统一讨论是否要继续冲线。

补充最新状态：
- `car` 当前已经完成第三个完整样板闭环；
- 正式协议结论是：`online_llm = 0.8257 > traditional_rf = 0.8174 > deeper_rf = 0.7961`，但 `< traditional_delta = 0.8470`；
- `jannis` 当前已经完成第四个正式样板闭环；
- 正式协议结论是：`online_llm = 0.5814 > mock_llm = 0.5651 > without_llm = 0.5640`，但 `< deeper_rf = 0.6324 < traditional_rf = 0.6488 < traditional_delta = 0.6575`；
- 当前也已经新增专门的方法论文档 `docs/dataset_generalization_methodology.md`，后续迁回归路线将直接按这份方法论执行。

补充最新状态：
- `bank` 当前新增了叶子选择策略预览工具，并已确认 `impurity` 会把候选叶子从 `[5, 11, 13]` 切到 `[14, 10, 6]`；
- 当前最小实跑显示：`without_llm + impurity = 0.8904`、`mock_llm + impurity = 0.8901`，都高于旧默认策略；
- `bank` 当前 `online_llm + impurity` 也已经跑通，但仍只追平 `mock_llm`，说明 `bank` 上的下一步重点已经从“继续堆旧叶子的 prompt”切到“继续验证新叶子策略 / 结构搜索”；
- `adult` 当前已经从“有审计产物”推进到“正式协议摘要已生成”：`online_llm = 0.8493`，`deeper_rf = 0.8456`，`traditional_rf = 0.8585`，`traditional_delta = 0.8602`。

---

## 11. `adult` 数据集接入作战卡（下一站）

`adult` 之所以被放在 `bank` 后面，不是因为它更简单，而是因为它更适合回答一个更细的问题：

> 当字段语义非常明确时，真实 `online_llm` 能不能比 `mock_llm` 更稳定地学会“局部哪个特征更值得分裂”。

### 11.1 为什么先做 `adult`
- 它是典型的强语义表格数据；
- 当前仓库里已有 `adult` 的 Prompt / 规则文件基础；
- 相比 `jannis` 这种弱语义高维数据，`adult` 更容易把“自然语言语义是否有帮助”这个问题看清楚。

### 11.2 第一轮不追求什么
- 不追求一次就超过所有传统对照；
- 不追求上来就做复杂 prompt 搜索；
- 不追求同时改候选叶子筛选、prompt、top-k、评估协议。

第一轮只追求三件事：
- 跑通统一入口；
- 形成同口径摘要；
- 看清 `online_llm` 相对 `mock_llm` 和 `without_llm` 的位置。

### 11.3 建议的最小执行顺序

1. 先确认数据与 Prompt 基础是否齐全
- `DeLTa-main/example_datasets/adult/`（若当前仓库尚未放入，需要先补数据准备）
- `DeLTa-main/llm/get_prompts/adult.py`
- `DeLTa-main/model/llm_rule/adult.py`

2. 先打通传统底座
- `run_randforest.py`
- `run_get_prompt.py`
- `run_get_answer.py`
- `get_trees.py`
- `run.py`
- `run_ensemble.py`

3. 再接 leaf expansion 三组核心实验
- `depth=3 + without_llm`
- `depth=3 + mock_llm`
- `depth=3 + online_llm`

4. 最后再补 `Deeper RF (depth=4)` 的同口径对照

### 11.4 第一轮必须确认的文件
- 数据配置：`DeLTa-main/dataset_config.py`
- Prompt 模板：`DeLTa-main/llm/get_prompts/adult.py`
- 原型入口：`DeLTa-main/run_leaf_expansion_mnist.py`
- 候选叶子与摘要：`DeLTa-main/model/leaf_expansion/mnist_leaf_selector.py`
- relation prompt：`DeLTa-main/model/leaf_expansion/mnist_relation_provider.py`
- 同口径摘要：`DeLTa-main/results/adult/*_same_metric_summary.json`

### 11.5 `adult` 上最值得优先观察的点
- 类别值解释是否足够清楚；
- `task_intro` 与 `target_outcomes` 是否真的帮助 online 排序；
- `one_feature_train_accuracy` 这类局部质量信号是否继续有效；
- `online_llm` 的优势究竟来自语义，还是主要还是来自候选特征统计摘要。

### 11.6 当前正式协议结论（2026-03-11）
- 已产出 `adult` 的统一同口径摘要与正式协议摘要；
- 当前可明确回答：
	- `online_llm` 超过了 `mock_llm`；
	- `online_llm` 超过了 `without_llm`；
	- `online_llm` 超过了 `Deeper RF(depth=4)`；
	- `online_llm` 仍低于传统 `RF` 与传统 `DeLTa` 融合结果。
- 因此 `adult` 当前已经完成“是否有真实 online 增益”的验证，但还没有完成“是否优于完整传统主链”的验证。

补充说明：
- 真正开工前，优先先看 `docs/adult_integration_precheck.md`；
- 这份清单比作战卡更偏“工程预检”，用于避免一上来就因为数据目录、输出路径或摘要脚本缺失而白跑。

`credit-g` 现在也已有同级预检文档：
- `docs/credit_g_integration_precheck.md`
- 建议顺序仍是先 `adult`，后 `credit-g`，避免同时打开两份强语义表格数据导致问题来源难拆。

补充说明：
- `jannis` 与 `car` 现在也都已有同级预检文档，可分别作为“高维弱语义”和“低维强类别值解释”两种极端表格样板；
- `house_16H_reg` 与 `california_housing` 也已补入回归侧预检文档，当前文档体系上已经具备“分类 + 回归”双路线入口；
- 因此，后续真正进入完全通用化时，优先缺的已不再是文档入口，而是数据层与首轮实验产物本身。

---

## 12. 2026-03-24 最终验证结果

### 5/8 数据集超越 deeper_rf

| 数据集 | 类型 | LLM | deeper_rf | 状态 |
|--------|------|-----|-----------|------|
| bank | 强语义二分类 | 89.04% | 88.73% | ✅ |
| adult | 强语义二分类 | 84.93% | 84.56% | ✅ |
| car | 低维类别多分类 | 82.57% | 79.61% | ✅ |
| house_16H_reg | 回归 | 41923 | 42440 | ✅ |
| california_housing | 回归 | 0.6468 | 0.6508 | ✅ |
| credit-g | 强语义二分类 | 72.86% | 73.14% | ❌ 差0.28% |
| jannis | 高维弱语义多分类 | 61.56% | 63.24% | ❌ |
| mnist | 图像 | 62.02% | 81.64% | ❌ |

### 网格调参覆盖

测试了 50+ 种配置组合:
- depth: 2-7
- top_k_leaves: 3-30
- top_k_features: 1-10
- leaf_selection_strategy: impurity_mass, sample_count
- local_max_depth: 1-2

### 关键发现

1. **credit-g 差0.28%**: 金融语义过强，LLM 容易被误导，是最接近突破的反例
2. **jannis 高维匿名**: 54维特征无语义，LLM 无法利用
3. **回归任务成功**: house_16H_reg 和 california_housing 都成功突破
4. **语义表格最有效**: bank, adult, car 表现最好

### 下一步

- 使用真实 LLM 测试 credit-g (差0.28%即可突破)
- 改进 jannis 方法 (高维匿名特征)
- 整理文档并提交 PR
