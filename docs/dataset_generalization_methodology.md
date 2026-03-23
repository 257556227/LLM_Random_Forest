# 数据集通用化方法论（第三数据集闭环后固化版）

## 1. 这份方法论文档解决什么问题

这份文档不是单个数据集的实验记录，而是把当前已经跑过的 `MNIST / bank / adult / credit-g / car` 经验，固化成一套可以继续迁到 `jannis / house_16H_reg / california_housing` 的工程方法。

目标只有一个：

> 新开一个数据集时，不再靠临场发挥，而是按固定步骤判断、接入、调优、收口。

---

## 2. 先给结论：后续数据集统一按 6 步走

### 第 1 步：先给数据集归类，再决定预期
不要一上来就跑代码，先判断它属于哪一类：

1. **强语义二分类表格**：`adult`、`credit-g`
2. **低维纯类别多分类表格**：`car`
3. **高维弱语义多分类表格**：`jannis`
4. **图像/弱自然语言语义特征**：`MNIST`
5. **回归表格**：`house_16H_reg`、`california_housing`

不同类型对应不同预期：
- 强语义数据重点看 `online_llm` 是否比 `mock_llm` 更会选局部特征；
- 低维纯类别数据重点看类别值解释是否足以带来稳定增益；
- 高维弱语义数据重点看局部统计摘要是否还能支撑 LLM 排序；
- 回归数据重点看 prompt 和评估口径能否先闭环。

### 第 2 步：先补数据准备，不要先写 prompt 调优
统一要求：
- 生成 `example_datasets/<dataset>/`
- 写全 `y_train/y_val/y_test.npy`
- 至少保证一类特征矩阵存在：`N_*` 或 `C_*`
- `info.json` 必须写清：
  - `task_type`
  - `num_classes`（分类）
  - `n_num_features` / `n_cat_features`
  - `task_intro`
  - `feature_intro`
  - `target_intro`
  - `source`

经验：
- `adult / credit-g` 的成败差异，很大一部分来自 `feature_intro + target_intro` 是否说清楚；
- `car` 证明了即使没有数值特征，只要类别值解释足够明确，也能形成稳定 online 增益。

### 第 3 步：先跑 readiness，再跑实验
先生成 readiness 报告：
- `tools/audit_dataset_integration_readiness.py --dataset <dataset>`

只有下面 4 类都满足，再进入真实实验：
- `dataset_config.py` 已配置
- `llm/get_prompts/<dataset>.py` 已存在
- `model/llm_rule/<dataset>.py` 已存在
- 数据目录和 `info.json` 已存在

这样做的价值是：
- 避免把“目录没准备好”误判成“算法没效果”；
- 避免后面文档、台账、模块映射又补漏一轮。

### 第 4 步：先跑三组原型，再决定值不值得补正式协议
统一先跑：
- `without_llm`
- `mock_llm`
- `online_llm`

为什么先跑这三组：
- `without_llm` 告诉我们“局部扩展本身”值不值得；
- `mock_llm` 告诉我们“稳定启发式排序”能到哪里；
- `online_llm` 才告诉我们真实语义有没有增益。

如果 `online_llm` 连 `mock_llm` 都追不上，先不要急着跑很重的 sweep；
先回答：
1. 是 prompt 语义太泛？
2. 是候选叶子选错了？
3. 还是这个数据集本身更依赖局部统计而不是自然语言语义？

### 第 5 步：正式协议只保留三组主对照
正式协议固定成：
- `Deeper RF (depth=4)`
- `depth=3 + without_llm`
- `depth=3 + online_llm`

辅助对照：
- `prototype_baseline`：depth=3 原型锚点
- `mock_llm`：sanity check
- `traditional_rf`
- `traditional_delta`

这样做的原因：
- `Deeper RF` 回答“传统直接加深有没有更划算”；
- `without_llm` 回答“局部结构扩展本身有没有价值”；
- `online_llm` 才回答“LLM 语义引导有没有额外价值”。

### 第 6 步：调优顺序固定，不允许乱扫
后续调优统一按这个顺序走：

1. **候选叶子策略**
   - 先试 `impurity_mass / impurity / sample_count`
   - 原因：`bank` 已证明“换叶子”常常比“修 prompt”更有效

2. **prompt 语义增强**
   - 优先补：`task_intro`、`target_outcomes`、`predicted_outcome`、`alternative_outcomes`
   - 再补：`category_values_preview`、`predicted_class_top_categories`
   - 再补：`one_feature_train_accuracy`

3. **局部搜索范围**
   - 调 `top_k_features_to_try`
   - 必要时调 `top_k_leaves`

4. **最后才考虑更大结构改动**
   - 如果前三步都无效，再考虑新结构或新搜索逻辑

---

## 3. 当前 5 个已跑样板带来的决策规则

### 3.1 `adult` 规则
- `online_llm > without_llm > mock_llm`
- `online_llm > Deeper RF`
- 但 `< traditional RF / DeLTa`

结论：
- 强语义字段确实能帮助 online 排序；
- 但要超过完整传统主链，还需要进一步缩小主链 gap。

### 3.2 `credit-g` 规则
- `mock_llm` 能保住基线
- `online_llm` 低于 `mock_llm / Deeper RF / RF / DeLTa`

结论：
- 强语义不等于 online 一定更强；
- 金融语义很容易把模型带离叶子局部最优分裂。

### 3.3 `car` 规则
- `online_llm = 0.8257`
- `without_llm = 0.8043`
- `mock_llm = 0.7911`
- `deeper_rf = 0.7961`
- `traditional_rf = 0.8174`
- `traditional_delta = 0.8470`

结论：
- 低维纯类别多分类数据上，类别值解释非常有效；
- `online_llm` 已经超过 `Deeper RF` 和传统 `RF`；
- 但仍低于传统 `DeLTa` 融合主链；
- 调优上，`impurity_mass` 与 `impurity` 等价，`sample_count` 更差，说明该类数据不该优先按样本数选叶子。

### 3.4 `bank` 规则
- online 能追平 mock
- 但继续堆旧叶子 prompt 收益有限

结论：
- 当在线排序已经逼近 mock 时，优先做候选叶子和结构搜索，不要继续打磨措辞。

### 3.5 `MNIST` 规则
- online 略高于 mock
- 但仍远低于 Deeper RF

结论：
- 图像类特征对 richer prompt 和 top-k 有响应；
- 但当前方法增益仍不足以替代传统容量提升。

### 3.6 `jannis` 规则
- `online_llm = 0.5814`
- `without_llm = 0.5640`
- `mock_llm = 0.5651`
- `deeper_rf = 0.6324`
- `traditional_rf = 0.6488`
- `traditional_delta = 0.6575`

结论：
- 高维弱语义多分类上，online 仍能从局部统计摘要里学到有效排序，且明显超过 `mock_llm / without_llm / baseline`；
- 但它和传统容量提升之间仍有明显 gap，说明这类数据目前更像“online 对原型有效，但不足以替代更强树容量”的样板；
- 调优上，默认 `impurity_mass` 依旧最好，切到 `impurity` 反而明显下降，说明这类数据更依赖覆盖高杂质质量而不是单纯换叶子。

---

## 4. 迁到新数据集时的最小文件清单

每开一个新数据集，至少要补这 8 类：
1. `tools/prepare_<dataset>_dataset.py`
2. `results/<dataset>/<dataset>_integration_readiness.json`
3. `tools/evaluate_<dataset>_leaf_expansion_same_metric.py`
4. `tools/evaluate_<dataset>_formal_protocol.py`
5. `experiments/<dataset_slug>_experiment_ledger.md`
6. `DeLTa-main/docs/module_map_<dataset_slug>.md`
7. `test/test_prepare_<dataset_slug>_dataset.py`
8. `test/test_<dataset_slug>_integration.py`

这 8 类补齐，才算这个数据集真正进入“可维护状态”。

---

## 5. 后续数据集的建议顺序

## 5.1 当前主线重置：先扩满 8 个数据集，再讨论单点极限追分

从当前阶段开始，主线不再是“继续围绕单个数据集死磕是否立刻超过 `deeper_rf`”，而是：

1. 先把当前 8 个候选数据集都推进到“有 readiness / 三组原型 / 正式协议 / 台账 / 测试”的可维护状态；
2. 在扩展过程中持续记录哪些数据集已经 `online_llm > deeper_rf`，哪些仍然做不到；
3. 如果某类数据集在完成高收益动作后依然明显低于传统对照，再把它正式归类为“当前阶段无法稳定超过”的反例样板，而不是继续无上限细磨。

这样做的价值是：
- 先回答“这套方法在哪些数据类型上有效、在哪些类型上无效”；
- 避免在单个反例数据集上投入过多时间，拖慢整条 8 数据集通用化主线；
- 让后续是否继续追 `deeper_rf`，建立在完整样板分布上，而不是建立在单点直觉上。

## 5.2 已验证的高收益动作优先级

后续无论推进哪个数据集，都优先做下面这些高收益动作；如果它们做完仍然无效，再接受“当前比不了”的结论：

1. **先补齐数据层与 readiness**
   - 没有 `example_datasets/<dataset>/`、`info.json`、`readiness`，就不要讨论算法优劣；
   - 这是所有后续结论可信的前提。

2. **先跑三组原型，不先做大 sweep**
   - 固定 `without_llm / mock_llm / online_llm`；
   - 先判断问题出在 LLM 语义、候选叶子，还是结构容量不够。

3. **候选叶子策略先于 prompt 微调**
   - 先试 `impurity_mass / impurity / sample_count`；
   - `bank` 已证明“换叶子”经常比“修旧 prompt 措辞”更值钱。

4. **补局部质量信号先于堆长 prompt**
   - 优先补 `task_intro`、`target_outcomes`、`predicted_outcome`、`alternative_outcomes`、`one_feature_train_accuracy`；
   - 这是目前跨 `bank / adult / car / jannis` 都验证过更稳的动作。

5. **扩大局部结构搜索优先于低收益超参细磨**
   - 先试 `top_k_features_to_try`、`top_k_leaves`；
   - 再试更深局部树；
   - 再试多特征局部结构搜索；
   - 当前 `jannis` 已证明“双特征局部树”比“更深单特征树”更有收益。

6. **完成正式协议后及时收口**
   - 如果高收益动作做完，结果仍明显低于 `deeper_rf / traditional_rf / traditional_delta`，就把它登记为正式反例；
   - 不要无限期停留在单一数据集上。

### 下一站：`house_16H_reg`
原因：
- `adult / credit-g / car / jannis` 已经覆盖了强语义二分类、低维纯类别多分类和高维弱语义多分类；
- 当前分类路线的通用化样板已经够多，最缺的是回归口径闭环；
- `house_16H_reg` 最适合作为第一个回归样板，去验证 prompt、评估口径和结果落盘是否也能复用当前流程。

### 再下一站：`california_housing`
原因：
- 需要第二个回归数据集确认第一份回归方法论不是偶然有效；
- `california_housing` 能补上更常见的中等维度连续特征场景。

---

## 6. 一句话工作流（以后照这个做）

> 先分类数据集类型，再补数据层和 readiness；先跑 `without/mock/online` 三组原型，再决定是否补正式协议；调优优先改候选叶子策略，其次改 prompt 语义，再次调局部搜索范围；最后把结论写回台账、主入口和测试。

---

## 7. 2026-03-24 最终验证结果

### 5/8 数据集超越 deeper_rf

| 数据集 | 类型 | LLM最佳 | deeper_rf | 状态 |
|--------|------|---------|-----------|------|
| bank | 强语义二分类 | 89.04% | 88.73% | ✅ |
| adult | 强语义二分类 | 84.93% | 84.56% | ✅ |
| car | 低维类别多分类 | 82.57% | 79.61% | ✅ |
| house_16H_reg | 回归 | 41923 | 42440 | ✅ |
| california_housing | 回归 | 0.6468 | 0.6508 | ✅ |
| credit-g | 强语义二分类 | 72.86% | 73.14% | ❌ 差0.28% |
| jannis | 高维弱语义多分类 | 61.56% | 63.24% | ❌ |
| mnist | 图像 | 62.02% | 81.64% | ❌ |

### 经验总结

1. **强语义表格数据最有效**: bank, adult, car 都能超越
2. **回归任务可行**: house_16H_reg, california_housing 成功突破
3. **金融语义反例**: credit-g 因语义过强导致 LLM 被误导
4. **高维匿名特征困难**: jannis, mnist 无法利用语义信息
