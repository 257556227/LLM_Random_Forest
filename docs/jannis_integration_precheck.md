# jannis 数据集接入前检查清单

## 1. 这份清单解决什么问题

这份文档专门回答：

> 如果在 `adult / credit-g` 之后继续推进 `jannis`，有哪些底座已经具备，哪些缺口还没补，第一轮该怎么稳妥落地。

它的目标是把 `jannis` 从“高维候选数据集”变成“可执行的下一阶段对象”。

---

## 2. 当前核查结论（2026-03-11）

### 2.1 已具备
- `DeLTa-main/dataset_config.py` 已有 `jannis` 参数配置。
- `DeLTa-main/llm/get_prompts/jannis.py` 已存在。
- `DeLTa-main/model/llm_rule/jannis.py` 已存在。
- `DeLTa-main/run_leaf_expansion_mnist.py` 的 `--dataset` 参数是通用的，可复用到 `jannis`。
- `DeLTa-main/model/leaf_expansion/` 这一层逻辑都可复用。

### 2.2 当前缺失
- `DeLTa-main/example_datasets/jannis/` 已真实生成。
- `DeLTa-main/results/jannis/` 已有首轮 leaf expansion 与正式协议产物。
- `jannis` 的 leaf expansion 同口径摘要脚本与结果文件已补齐。
- `jannis` 的实验台账与模块映射文档已补齐。

### 2.3 需要特别注意
- `jannis` 是高维多分类表格数据，语义通常弱于 `adult` / `credit-g`。
- 这意味着第一轮更适合验证“候选特征摘要 + top-k 机制是否还能稳定”，而不是过早强调自然语言语义本身。
- 真正跑时必须显式覆盖输出路径，避免仍落到 `results/mnist/...`。

---

## 3. 开工前必须完成的检查

### 3.1 数据层检查
- [x] 已确认存在 `DeLTa-main/example_datasets/jannis/`
- [x] 已准备：
  - [x] `N_train.npy / N_val.npy / N_test.npy`
  - [x] `y_train.npy / y_val.npy / y_test.npy`
  - [x] `info.json`
- [x] `info.json` 已包含：
  - [x] `task_type`
  - [x] `n_num_features`
  - [x] `n_cat_features`
  - [x] `task_intro`
  - [x] `feature_intro`
  - [x] `target_intro`
  - [x] `source`

### 3.2 传统 DeLTa 底座检查
- [x] `DELTA_DATASETS=jannis` 已跑通 `run_randforest.py`
- [ ] `run_get_prompt.py` 能生成 `jannis` prompt
- [ ] `run_get_answer.py` 能落出 `jannis` answers（若在线环境不可用，要记录阻塞）
- [ ] `get_trees.py` 能把答案转成规则
- [x] `run.py` 已基于现有 `model/llm_rule/jannis.py` 跑通训练
- [x] `run_ensemble.py` 已融合并产生日志

### 3.3 Leaf Expansion 原型检查
- [x] `python run_leaf_expansion_mnist.py --dataset jannis --mode without_llm ...` 已跑通
- [x] `python run_leaf_expansion_mnist.py --dataset jannis --mode llm_guided --relation_source mock_llm ...` 已跑通
- [x] `python run_leaf_expansion_mnist.py --dataset jannis --mode llm_guided --relation_source online_llm ...` 已正式尝试并落盘
- [x] 已显式覆盖输出路径到 `results/jannis/...`

### 3.4 文档与产物检查
- [x] 已新增 `experiments/jannis_experiment_ledger.md`
- [x] 已新增 `DeLTa-main/docs/module_map_jannis.md`
- [x] 已在 `Update.md` 中记录本轮尝试
- [x] 已把 `jannis` 结果补进完全通用化文档

---

## 4. 第一轮推荐执行顺序
- 先补数据层
- 再跑传统底座
- 再跑 `without_llm` 与 `mock_llm`
- 最后开 `online_llm`
- 再补 `Deeper RF` 同口径对照

---

## 5. 当前最可能遇到的阻塞
- 高维弱语义特征确实限制了最终 ceiling：`online_llm` 明显强于三组原型，但仍低于 `Deeper RF / RF / DeLTa`
- 即便 prompt 更丰富，当前主要增益仍来自局部统计摘要而不是强语义字段
- 当前统一摘要已经补齐，后续可以直接和 `adult / credit-g / car` 横向比较

## 5.1 当前正式协议结论
- `prototype_baseline = 0.5667`
- `without_llm = 0.5640`
- `mock_llm = 0.5651`
- `online_llm = 0.5814`
- `deeper_rf = 0.6324`
- `traditional_rf = 0.6488`
- `traditional_delta = 0.6575`

这说明：
- `jannis` 当前已经是一个正向原型样板；
- `online_llm` 明显超过 `mock_llm / without_llm / prototype_baseline`；
- 但它仍没有超过 `Deeper RF / traditional RF / traditional DeLTa`。

## 5.2 当前调优结论
- `impurity` 会把结果压到 `0.5680` 左右，明显劣于默认；
- `sample_count`、`top_k_leaves=4`、`top_k_features_to_try=4` 都没有超过默认 `0.5814`；
- 因此默认 `impurity_mass + top_k_leaves=3 + top_k_features_to_try=3` 仍是当前最优收口点。

---

## 6. 第一轮完成标准
- [x] 数据目录完整
- [x] DeLTa 传统底座可跑
- [x] `without_llm` 可跑
- [x] `mock_llm` 可跑
- [x] `online_llm` 已正式尝试并落盘
- [x] 已有 `jannis` 的统一同口径摘要与正式主对比协议

## 7. 当前工程判断

- `jannis` 当前已经完成“高维弱语义多分类样板”的完整闭环；
- 它证明 online 语义引导对原型确实有价值，但当前还不足以替代传统容量提升；
- 下一步最值得转向回归路线，而不是继续在 `jannis` 上做低收益细磨。
