# car 数据集接入前检查清单

## 1. 这份清单解决什么问题

这份文档专门回答：

> 如果后续要用 `car` 作为低维强类别值解释样板，真正开工前需要先确认哪些底座已经具备、哪些问题要提前规避。

---

## 2. 当前核查结论（2026-03-11）

### 2.1 已具备
- `DeLTa-main/dataset_config.py` 已有 `car` 参数配置。
- `DeLTa-main/llm/get_prompts/car.py` 已存在。
- `DeLTa-main/model/llm_rule/car.py` 已存在。
- `DeLTa-main/run_leaf_expansion_mnist.py` 的 `--dataset` 参数可复用到 `car`。
- `DeLTa-main/model/leaf_expansion/` 逻辑可复用。

### 2.2 当前缺失
- `DeLTa-main/example_datasets/car/` 已真实生成。
- `DeLTa-main/results/car/` 已有首轮 leaf expansion 与正式协议产物。
- `car` 的 leaf expansion 同口径摘要脚本与结果文件已补齐。
- `car` 的实验台账与模块映射文档已补齐。

### 2.3 需要特别注意
- `car` 是低维、纯类别型、多分类数据；
- 这类数据最适合测试“类别值解释、target-conditioned prompt、counterfactual 提示”是否真的有用；
- 但也要警惕：传统树在低维数据上通常已经很强，留给叶子扩展的空间可能很小。

---

## 3. 开工前必须完成的检查

### 3.1 数据层检查
- [x] 已确认存在 `DeLTa-main/example_datasets/car/`
- [x] 已准备：
  - [x] `C_train.npy / C_val.npy / C_test.npy`
  - [x] `y_train.npy / y_val.npy / y_test.npy`
  - [x] `info.json`
- [x] `info.json` 已包含：
  - [x] `task_type`
  - [x] `n_cat_features`
  - [x] `task_intro`
  - [x] `feature_intro.cat`
  - [x] `target_intro`
  - [x] `source`

### 3.2 传统底座检查
- [x] `DELTA_DATASETS=car` 已跑通 `run_randforest.py`
- [ ] `run_get_prompt.py` 能生成 `car` prompt
- [ ] `run_get_answer.py` 能落出 `car` answers
- [ ] `get_trees.py` 能把答案转成规则
- [x] `run.py` 已基于现有 `model/llm_rule/car.py` 跑通训练
- [x] `run_ensemble.py` 已融合并产生日志

### 3.3 Leaf Expansion 原型检查
- [x] `without_llm` 可跑
- [x] `mock_llm` 可跑
- [x] `online_llm` 已正式尝试并落盘
- [x] 已显式覆盖输出路径到 `results/car/...`

### 3.4 文档与产物检查
- [x] 已新增 `experiments/car_experiment_ledger.md`
- [x] 已新增 `DeLTa-main/docs/module_map_car.md`
- [x] 已在 `Update.md` 中记录本轮尝试
- [x] 已把 `car` 结果补进完全通用化文档

---

## 4. 第一轮最值得观察的点
- 类别值解释是否足以帮助 online 做局部排序
- `predicted_outcome` / `alternative_outcomes` 是否比数值型表格数据更有效
- online 是否只是追平 `mock_llm`，还是能真正超过它

## 4.1 当前正式协议结论
- `prototype_baseline = 0.7632`
- `without_llm = 0.8043`
- `mock_llm = 0.7911`
- `online_llm = 0.8257`
- `deeper_rf = 0.7961`
- `traditional_rf = 0.8174`
- `traditional_delta = 0.8470`

这说明：
- `car` 当前已经是一个正向样板；
- `online_llm` 不仅超过了 `mock_llm`，也超过了 `Deeper RF` 和传统 `RF`；
- 但它还没有超过传统 `DeLTa` 融合主链。

## 4.2 当前调优结论
- `impurity_mass` 与 `impurity` 在当前配置下效果相同；
- `sample_count` 会把候选叶子切到纯叶子，导致只扩出 1 个叶子，效果变差到 `0.8125`；
- 增加 `top_k_leaves` 到 `4`、增加 `top_k_features_to_try` 到 `4` 都没有进一步提升；
- 因此默认配置已经是当前最优收口点。

---

## 5. 第一轮完成标准
- [x] 数据目录完整
- [x] DeLTa 传统底座可跑
- [x] `without_llm` 可跑
- [x] `mock_llm` 可跑
- [x] `online_llm` 已正式尝试并落盘
- [x] 已有 `car` 的统一同口径摘要与正式主对比协议

## 6. 当前工程判断

- `car` 当前已经完成“第三个数据集完整闭环 + 调优 + 正式协议”；
- 它是目前最强的“低维纯类别多分类正向样板”；
- 下一步更值得把这套方法复制到 `jannis`，验证高维弱语义多分类是否也能复用，而不是继续在 `car` 上做低收益微调。
