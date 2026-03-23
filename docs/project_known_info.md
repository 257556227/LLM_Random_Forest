# 项目已知信息沉淀

## 1. 当前项目的真实边界

### 1.1 已经完成的部分
- 已完成 `DeLTa` 主流程复现：规则抽取 -> Prompt 生成 -> 在线查询 -> 规则落地 -> 误差修正训练 -> 融合评估。
- 已在 `MNIST` 上跑通单数据集闭环。
- 已完成 Prompt A / B / C 的同口径对照实验。
- 已形成较完整的工程底座：日志、台账、测试、更新记录、模块映射。
- 已完成 `MNIST` 上 `LLM-Guided Leaf Expansion Forest` 的最小可运行原型，并已跑通 `without_llm / mock_llm / json / online_llm` 四类 relation 口径。
- 已完成 `bank` 的第一轮迁移验证与 online relation 迁移验证，当前 `bank online_llm` 已追平 `mock_llm`。

### 1.2 目前还没有完成的部分
- 当前版本已经进入“LLM 直接参与叶子局部结构生成”的原型阶段，但**还没有完成完全通用化**。
- 当前还不能宣称已经在多数据集上稳定证明“新机制优于传统更深树”。
- 当前距离正式方法结论还差三步：
  - 把主对比协议固定为同一底座下的结构比较；
  - 把 `bank` 从“能迁移”推进到“可持续调优”；
  - 把方法扩到更多公开数据集，并争取做到 `> mock_llm` 且 `> 传统对照`。

## 2. 后续方法的正式定位

建议后续统一命名为以下二选一：
- `Relation-Aware Random Forest`
- `LLM-Guided Leaf Expansion Forest`

更推荐第二个名称，因为它更准确描述当前准备落地的机制：
- 不改 RF 前部主干；
- 只在终端或近终端节点增加一个局部扩展节点；
- 新增 split 的候选特征由 LLM 识别的 feature relation 提供；
- threshold 仍由本地数据上的 CART / stump 计算，而不是让 LLM 直接决定阈值。

## 3. 建议的方法边界

### 3.1 这不是 DeLTa Prompt 改进版
后续工作不应再表述为：
- “基于 DeLTa 改 Prompt”；
- “继续优化 Prompt 让结果更好”。

更准确的表述应当是：
- 以 `DeLTa` 为可运行基线；
- 将 LLM 的角色从“后修正 / 后融合”前移为“局部结构先验提供者”；
- 进入树生成机制本身，而不是停留在最后融合。

### 3.2 这也不是一开始就做 tree-to-graph
- `tree-to-graph` 更适合作为扩展方向；
- 第一版建议只做 `leaf expansion`；
- 先证明局部结构增强是否有效，再决定是否进一步转向 graph 结构。

## 4. 推荐的统一实验协议

### 实验 1：Depth-3 RF（内部锚点，不作为主报告重点）
- `depth = 3`
- `n_trees = 100`

这组实验继续保留，但后续不再作为主要汇报对比对象，而是作为所有叶子扩展实验共享的“同一底座”。

### 实验 2：Deeper RF（主对照之一）
- `depth = 4`
- `n_trees = 100`

### 实验 3：Leaf Expansion without LLM（主对照之一）
- 前 3 层照常生长；
- 对候选叶子额外补 1 层局部扩展；
- 不使用 LLM relation，只使用普通 impurity 选择。

这组实验和 `depth = 4` 不是一个意思：
- `Deeper RF` 是**整棵树统一多长一层**；
- `without_llm` 是**保持 3 层底座不变，只对少数候选叶子做局部补 1 层**。

因此，`Deeper RF` 回答的是“传统上直接加深是否更强”；
而 `without_llm` 回答的是“只做局部结构扩展但不用 LLM，单靠局部 impurity 选择有没有收益”。

### 实验 4：mock_llm（可选 sanity check，不再作为主优先级）
- 前 3 层照常生长；
- 对候选叶子额外补 1 层局部扩展；
- 候选特征排序由 `mock_llm` 这种确定性启发式给出。

这组实验后续仍保留，但定位改为：
- 用来判断真实 `online_llm` 是否确实优于“无语义的稳定假基线”；
- 不再作为主要汇报表的首要比较对象。

### 实验 5：online LLM-Guided Leaf Expansion（主对照之一）
- 前 3 层照常生长；
- 对候选叶子额外补 1 层局部扩展；
- 由真实 `online_llm` 给出候选 feature 排序；
- threshold 由本地 stump 计算。

### 后续主汇报建议
后续正式汇报优先固定三组主对照：
- `Deeper RF (depth=4)`
- `depth=3 + without_llm`
- `depth=3 + online_llm`

其中：
- `depth=3` 是公共底座锚点；
- `mock_llm` 是辅助 sanity check；
- 真正要回答的问题是：
  - `online_llm` 是否优于“传统直接加深一层”；
  - `online_llm` 是否优于“没有语义的局部扩展”。

### 统一记录指标
- Accuracy
- AUC
- 训练时间
- 推理时间
- 扩展叶子数
- 总 split 数 / 平均复杂度

## 5. 当前已有关键结论

### 5.1 Prompt 对照结论
在 `MNIST` + 固定参数口径下：
- Prompt A = `0.9198`
- Prompt B = `0.9154`
- Prompt C = `0.9153`

当前 `DeLTa` 线上的最佳模板仍是 Prompt A。

### 5.2 负结果同样重要
当前已观察到一个重要负结果：
- `LeafExpansion` 的现有原型版本不如 `Deeper RF`。

这个结果不能简单理解为“方向错误”，更合理的解释包括：
- relation_map 质量不够高；
- 叶子筛选条件不稳定；
- 扩展策略可能把低价值叶子也扩展了；
- 把 `树更深` 与 `LLM 真有效` 混在一起比较，导致难以识别真正增益来源。

### 5.3 当前阶段判断已经变化
- `stage1_mnist_semigeneral_guide.md` 可以视为已完成；
- 当前主要阶段已进入 `stage1_bank_migration_guide.md`：重点不再是“能不能迁”，而是“能不能把迁移后的 online 继续做强”；
- 再往后的一阶段，不再叫“半通用化”，而是“调优 + 完全通用化”：
  - 在更多公开数据集上复现同一协议；
  - 尽量做到 `online_llm > mock_llm`；
  - 尽量做到 `online_llm > 传统对照`。

## 6. 关于公共数据集可获得性

大多数对标 `DeLTa` / `TnT` 的常见公开数据集理论上都可以获得，主要来源通常包括：
- OpenML
- UCI
- LIBSVM / 公开镜像
- 论文项目仓库附带数据链接

这意味着：
- 以 `MNIST` 跑通后，后续扩展到其他公共数据集在工程流程上通常是类似的；
- 但仍需做数据格式适配、指标适配和参数适配，不能假设“换数据集零成本”。

## 7. 当前建议的工程组织方式

建议正式版至少包含这些部分：
- `baseline/`：普通 RF 与 deeper RF
- `relation/`：特征关系提取
- `leaf_expansion/`：末端扩展逻辑
- `experiments/`：实验配置和结果记录
- `docs/`：方法说明、实验设计、失败结论

## 8. 后续最优先事项
- 已完成方法命名、统一入口与 `MNIST + bank` 跑通；
- 当前最优先的是：
  - 固定新的主对比协议（`Deeper RF / without_llm / online_llm`）；
  - 把 `bank` 从“已追平 mock”继续推进到“尽量超过 mock 与传统对照”；
  - 以 `docs/mnist_llm_guided_leaf_expansion_prototype.md` 为母版，扩展到更多公开数据集；
  - 把通用化过程中的适配点、失败模式、成功经验持续沉淀到单独 markdown 中。

## 9. 完全通用化目标（下一阶段）

后续正式目标不再只是“`MNIST + 1个 bank` 跑通”，而是：
- 以 `MNIST` 原型为母版；
- 推进到不少于 `8` 个公开数据集；
- 在每个数据集上尽量形成统一同口径对照；
- 对每个数据集优先看三件事：
  - `online_llm` 是否超过 `without_llm`；
  - `online_llm` 是否超过 `mock_llm`；
  - `online_llm` 是否超过传统主对照（优先看 `Deeper RF`）。

如果某些数据集不能三项都赢，也必须把原因写清楚：
- 是语义提示问题；
- 是候选叶子筛选问题；
- 是数据类型适配问题；
- 还是方法本身在该数据集上不占优。
