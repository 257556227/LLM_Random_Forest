# california_housing 数据集接入前检查清单

## 1. 这份清单解决什么问题

这份文档专门回答：

> 如果把当前机制推进到第二个回归型表格数据集，`california_housing` 在真正开始接入前，工程底座、Prompt/规则侧、数据层与实验收口层分别处于什么状态。

---

## 2. 当前核查结论（2026-03-10）

### 2.1 已具备
- `DeLTa-main/dataset_config.py` 已有 `california_housing` 配置。
- `DeLTa-main/llm/get_prompts/california_housing.py` 已存在。
- `DeLTa-main/model/llm_rule/california_housing.py` 已存在。
- `DeLTa-main/run_leaf_expansion_mnist.py` 已支持通用 `--dataset` 入口，理论上可复用。

### 2.2 当前缺失
- `DeLTa-main/example_datasets/california_housing/` 当前不存在。
- `DeLTa-main/results/california_housing/` 当前不存在。
- 还没有 `california_housing` 的 leaf expansion 实验台账、模块映射和同口径摘要。

### 2.3 需要特别注意
- 这是回归任务，和 `house_16H_reg` 一样，会直接检验当前机制对连续目标的适配能力；
- 但与 `house_16H_reg` 不同，这个数据集当前连本地数据目录都还未落地，因此它不是“可立即试跑”，而是“先补数据层，再谈实验层”；
- 如果 `house_16H_reg` 是回归适配样板，那么 `california_housing` 更适合作为“第二回归数据集复验”。

---

## 3. 开工前必须完成的检查

### 3.1 数据层检查
- [ ] 建立 `DeLTa-main/example_datasets/california_housing/`
- [ ] 至少补齐：
  - [ ] `N_train.npy`
  - [ ] `N_val.npy`
  - [ ] `N_test.npy`
  - [ ] `y_train.npy`
  - [ ] `y_val.npy`
  - [ ] `y_test.npy`
  - [ ] `info.json`
- [ ] `info.json` 中补齐：
  - [ ] `task_type = regression`
  - [ ] `task_intro`
  - [ ] 特征说明
  - [ ] 目标变量解释
  - [ ] `source`

### 3.2 传统底座检查
- [ ] `DELTA_DATASETS=california_housing` 能跑 `run_randforest.py`
- [ ] `run_get_prompt.py` 能生成 prompt
- [ ] `run_get_answer.py` 能得到 answers
- [ ] `get_trees.py` 能生成规则
- [ ] `run.py` 能训练
- [ ] `run_ensemble.py` 能融合并落结果

### 3.3 Leaf Expansion 原型检查
- [ ] `--dataset california_housing` 入口可跑
- [ ] `without_llm` 可跑
- [ ] `mock_llm` 可跑
- [ ] `online_llm` 至少尝试并记录结果/阻塞
- [ ] 明确输出目录到 `results/california_housing/...`

### 3.4 文档与产物检查
- [ ] 新增 `experiments/california_housing_experiment_ledger.md`
- [ ] 新增 `DeLTa-main/docs/module_map_california_housing.md`
- [ ] 在 `Update.md` 记录接入与阻塞
- [ ] 在完全通用化文档中同步阶段状态

---

## 4. 当前最可能遇到的阻塞
- 最大阻塞不是 prompt，而是数据目录当前缺失
- 即便数据补齐，回归任务的 prompt、特征摘要与评估口径仍可能沿用分类叙述
- 若先做 `house_16H_reg`，这里应尽量复用回归适配经验，避免重复踩坑

---

## 5. 第一轮完成标准
- [ ] 数据目录完整可读
- [ ] 传统底座能跑
- [ ] `without_llm` 可跑
- [ ] `mock_llm` 可跑
- [ ] `online_llm` 至少正式尝试并落盘
- [ ] 有 `california_housing` 的统一同口径摘要或明确阻塞说明
