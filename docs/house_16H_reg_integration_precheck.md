# house_16H_reg 数据集接入前检查清单

## 1. 这份清单解决什么问题

这份文档专门回答：

> 如果要把当前机制从分类推进到回归，`house_16H_reg` 作为第一回归样板，在真正开工前哪些条件已经具备，哪些条件还需要先补。

---

## 2. 当前核查结论（2026-03-12）

### 2.1 已具备
- `DeLTa-main/dataset_config.py` 已有 `house_16H_reg` 参数配置。
- `DeLTa-main/example_datasets/house_16H_reg/` 已存在。
- `DeLTa-main/example_datasets/house_16H_reg/info.json` 已存在，且已包含 `task_type = regression` 等元信息。
- `DeLTa-main/llm/get_prompts/house_16H_reg.py` 已存在。
- `DeLTa-main/model/llm_rule/house_16H_reg.py` 已存在。
- `DeLTa-main/run_leaf_expansion_mnist.py` 入口通用，可复用到回归数据，但仍需确认评估口径是否已适配。

### 2.2 当前缺失
- `DeLTa-main/results/house_16H_reg/` 现已存在三组 leaf expansion 原型结果、同口径摘要与正式协议摘要。
- 已新增 `experiments/house_16H_reg_experiment_ledger.md` 与 `DeLTa-main/docs/module_map_house_16H_reg.md`。
- 当前不再缺正式协议产物，但需要把“无 CUDA 环境下 tabpfn 自动回退到 cart”的运行时背景一并记录，避免后续误读结果口径。

### 2.3 需要特别注意
- 这是回归任务，不是分类任务；
- 当前很多 prompt、摘要与评估语言仍偏分类，因此这份数据集最容易暴露“分类 bias”；
- 真正接入时，要优先确认损失、评估指标和 prompt 描述是否已经从“类别分离”改成“连续目标区间分离”。

---

## 3. 开工前必须完成的检查

### 3.1 数据层检查
- [x] `DeLTa-main/example_datasets/house_16H_reg/` 已存在
- [x] 核对 `info.json` 是否足够支撑回归版 prompt：
  - [x] `task_type = regression`
  - [x] `task_intro`
  - [x] 数值特征说明（当前由 `num_feature_intro` 旧格式兼容读取）
  - [x] 目标变量解释
  - [x] `source`

### 3.2 传统底座检查
- [x] `DELTA_DATASETS=house_16H_reg` 能跑 `run_randforest.py`
- [ ] `run_get_prompt.py` 能生成 `house_16H_reg` prompt
- [ ] `run_get_answer.py` 能落出 answers
- [ ] `get_trees.py` 能把答案转成规则
- [x] `run.py` 能训练（当前环境无 CUDA 时会把 `tabpfn` 自动回退到 `cart`）
- [x] `run_ensemble.py` 能融合并产生日志

### 3.3 Leaf Expansion 原型检查
- [x] 原型入口能直接处理回归 `y`
- [x] `without_llm` 可跑
- [x] `mock_llm` 可跑
- [x] `online_llm` 已正式尝试并落盘
- [x] 输出路径已显式覆盖到 `results/house_16H_reg/...`

### 3.4 文档与产物检查
- [x] 新增 `experiments/house_16H_reg_experiment_ledger.md`
- [x] 新增 `DeLTa-main/docs/module_map_house_16H_reg.md`
- [x] 在 `Update.md` 中记录第一轮尝试
- [x] 把 `house_16H_reg` 结果补进完全通用化文档

---

## 4. 当前最可能遇到的阻塞
- 回归任务的评估口径还没有像分类那样明确收口
- prompt 仍可能偏向“分类哪一类”，需要改成“预测值区间/误差降低”
- 原型入口虽然通用，但局部扩展和摘要逻辑可能默认按分类解释 `predicted_class`

---

## 5. 第一轮完成标准
- [x] 传统底座可跑
- [x] 回归版 `without_llm` 可跑
- [x] 回归版 `mock_llm` 可跑
- [x] 回归版 `online_llm` 已正式尝试并落盘
- [x] 已有 `house_16H_reg` 的统一同口径摘要
- [x] 已有 `house_16H_reg` 的正式协议摘要

## 6. 当前第一轮结果（2026-03-11）

- `prototype_baseline`: `RMSE = 43327.9766`, `R2 = 0.3317`
- `without_llm`: `RMSE = 42303.1098`, `R2 = 0.3630`
- `mock_llm`: `RMSE = 42843.3905`, `R2 = 0.3466`
- `online_llm`: `RMSE = 42044.3802`, `R2 = 0.3707`

当前结论：
- 第一个回归样板已经真实跑通；
- `online_llm` 当前优于 `mock_llm` 与 `without_llm`，说明回归版 prompt 和局部结构扩展链路已经具备正向收益；
- 正式协议补齐后，当前 `online_llm` 仍低于 `Deeper RF / traditional RF / traditional DeLTa(cart 回退)`，因此应把 `house_16H_reg` 记为“原型有效但主对比仍未翻盘”的回归正式反例。

## 7. 正式协议结果（2026-03-12）

- `deeper_rf`: `RMSE = 38943.8822`, `R2 = 0.4601`
- `without_llm`: `RMSE = 42303.1098`, `R2 = 0.3630`
- `online_llm`: `RMSE = 42044.3802`, `R2 = 0.3707`
- `traditional_rf`: `RMSE = 35230.4577`, `R2 = 0.5582`
- `traditional_delta(cart回退)`: `RMSE = 41434.7702`, `R2 = 0.3888`

补充说明：
- 当前环境无 CUDA，`run.py / run_ensemble.py` 会把回归主链中的 `tabpfn` 自动回退到 `cart`，以避免 CPU `tabpfn` 在本机内存不足时直接失败；
- 因此这轮正式协议已经可复现，但 `traditional_delta` 结果应理解为“当前环境下的稳定可跑口径”，不是 CUDA TabPFN 口径；
- 下一步主线应切到 `california_housing`，用第二个回归数据集验证这个正式反例是否具有普遍性。
