# adult 数据集接入前检查清单

## 1. 这份清单解决什么问题

这份文档不讨论方法好不好，而是专门回答：

> 如果下一站真的要开 `adult`，在动代码和跑实验之前，哪些入口已经具备，哪些东西还缺，哪些地方必须先补。

它的目标是把 `adult` 从“想做”变成“可以按清单开工”。

---

## 2. 当前核查结论（2026-03-11）

### 2.1 已具备
- `DeLTa-main/dataset_config.py` 已有 `adult` 参数配置。
- `DeLTa-main/llm/get_prompts/adult.py` 已存在。
- `DeLTa-main/model/llm_rule/adult.py` 已存在。
- `DeLTa-main/run_leaf_expansion_mnist.py` 的 `--dataset` 参数是通用的，不是硬编码只支持 `mnist` / `bank`。
- `DeLTa-main/model/leaf_expansion/` 这一层的候选叶子、候选特征摘要、relation prompt、局部扩展逻辑都可复用。
- `adult` 的传统 DeLTa 语义基础已经存在。
- `DeLTa-main/example_datasets/adult/` 已真实生成，包含 `N/C/y` 三组划分文件与 `info.json`。
- `DeLTa-main/tools/evaluate_adult_leaf_expansion_same_metric.py` 已存在，并已生成首轮同口径摘要。
- `adult` 第一轮原型结果已经跑通：`prototype_baseline=0.8411645476322093`、`without_llm=0.8467538848965052`、`mock_llm=0.8440513481972852`、`online_llm=0.8492721577298692`。

### 2.2 当前缺失
- 当前已经不缺数据目录、原型摘要脚本和首轮结果文件。
- 当前主要未完成项变成：`adult` 的传统 DeLTa 主链基线、`Deeper RF` 基线，以及纳入正式主对比协议的同口径汇总。
- 换句话说，`adult` 已从“接入前预检”进入“首轮原型完成，准备进入正式协议”的阶段。

### 2.3 需要特别注意
- `run_leaf_expansion_mnist.py` 虽然支持通用 `--dataset`，但默认输出路径仍是 `results/mnist/...`，所以真正跑 `adult` 时必须显式覆盖 `--output` 和 `--guided_output`。
- 当前同口径评估工具是按 `MNIST` 和 `bank` 分开写的，`adult` 很可能需要新增对应评估脚本或扩展现有脚本。

### 2.4 当前最真实的工程判断
- `adult` 已经跨过“数据层没落地”的最大阻塞；
- 当前最值得继续推进的，不再是准备数据，而是把 `adult` 纳入正式主对比协议；
- 第一轮结果已经给出正向信号：`online_llm > without_llm > mock_llm`，因此它是当前最值得优先补正式 baseline 的强语义表格数据集。

### 2.5 最新就绪审计结果（2026-03-11）
- 当前已新增审计工具：`DeLTa-main/tools/audit_dataset_integration_readiness.py`
- 当前产物：`DeLTa-main/results/adult/adult_integration_readiness.json`
- 审计结果进一步确认：
  - 已具备：`dataset_config.py`、`llm/get_prompts/adult.py`、`model/llm_rule/adult.py`、预检文档、`results/adult/`、`experiments/adult_experiment_ledger.md`、`DeLTa-main/docs/module_map_adult.md`、`example_datasets/adult/`；
  - 当前 `missing_items = []`；
  - 当前 `ready_for_first_run = true`。

这意味着：
- `adult` 现在最该做的不是“再补接入前说明”，而是补 `Deeper RF` 与传统主链基线；
- 当前文档、数据和原型工具已经足够，不需要继续追加纯预检文档。

---

## 3. 开工前必须完成的检查

### 3.1 数据层检查
- [x] 已确认存在 `DeLTa-main/example_datasets/adult/`
- [x] 已准备以下文件：
  - [x] `N_train.npy / N_val.npy / N_test.npy`
  - [x] `C_train.npy / C_val.npy / C_test.npy`
  - [x] `y_train.npy / y_val.npy / y_test.npy`
  - [x] `info.json`
- [x] 已确认 `info.json` 中至少包含：
  - [x] `task_type`
  - [x] `task_intro`
  - [x] `feature_intro`
  - [x] `target_intro`

建议直接按 `bank/info.json` 的增强版结构准备，而不是只放最小字段。至少推荐包含：
- [x] `task_type`
- [x] `num_classes`
- [x] `n_num_features`
- [x] `n_cat_features`
- [x] `train_size / val_size / test_size`
- [x] `task_intro`
- [x] `feature_intro.num`
- [x] `feature_intro.cat`
- [x] `target_intro`
- [x] `source`

### 3.2 传统 DeLTa 底座检查
- [ ] `DELTA_DATASETS=adult` 能跑 `run_randforest.py`
- [ ] `run_get_prompt.py` 能生成 `adult` prompt
- [ ] `run_get_answer.py` 能落出 `adult` answers（若当前在线环境不可用，要记录阻塞）
- [ ] `get_trees.py` 能把答案转成规则
- [ ] `run.py` 能训练
- [ ] `run_ensemble.py` 能融合并产生日志

### 3.3 Leaf Expansion 原型检查
- [x] `python run_leaf_expansion_mnist.py --dataset adult --mode without_llm ...` 已跑通
- [x] `python run_leaf_expansion_mnist.py --dataset adult --mode llm_guided --relation_source mock_llm ...` 已跑通
- [x] `python run_leaf_expansion_mnist.py --dataset adult --mode llm_guided --relation_source online_llm ...` 已正式尝试并落盘
- [x] 已明确覆盖输出路径：
  - [x] `--output results/adult/...`
  - [x] `--guided_output results/adult/...`

推荐第一轮直接用下面这种命令风格，而不是吃默认路径：
- [ ] `python run_leaf_expansion_mnist.py --dataset adult --mode without_llm --output results/adult/leaf_expansion_without_llm_summary.json --save_summary`
- [ ] `python run_leaf_expansion_mnist.py --dataset adult --mode llm_guided --relation_source mock_llm --guided_output results/adult/llm_guided_leaf_expansion_summary.json --save_summary`

### 3.4 文档与产物检查
- [x] 已新增 `experiments/adult_experiment_ledger.md`
- [x] 已新增 `DeLTa-main/docs/module_map_adult.md`
- [ ] 在 `Update.md` 中记录第一轮尝试
- [ ] 把 `adult` 结果补进完全通用化文档

---

## 4. 第一轮推荐执行顺序

### 第一步：先补数据，不碰 prompt
这一步已完成，`example_datasets/adult/` 数据目录已落地。

### 第二步：先跑传统底座
先回答：`adult` 在 DeLTa 原链条上是不是完整可跑的。

### 第三步：先跑三组叶子扩展原型
先跑：
- `without_llm`
- `mock_llm`
- `online_llm`

目的不是出最终结论，而是确认：
- 候选叶子能不能选出来；
- 候选特征摘要是否正常；
- 输出路径是不是已经切到 `results/adult/`。

### 第四步：补正式主对比基线
当前最该补的是：
- `Deeper RF`
- 传统 DeLTa / RF 主链 baseline

### 第五步：最后补正式协议汇总
把主对比协议补齐成：
- `Deeper RF`
- `without_llm`
- `online_llm`

---

## 5. 当前最可能遇到的阻塞

### 阻塞 1：正式主链 baseline 还没补
这已经取代“数据目录不存在”，成为当前最明确的缺口。

### 阻塞 2：`info.json` 不够支撑 tabular prompt
如果缺少 `task_intro`、`feature_intro`、`target_intro`，online prompt 质量会明显下降。

### 阻塞 3：输出路径还落到 `mnist`
这不是算法问题，而是实验管理问题；但如果不先修，后续结果很容易混乱。

### 阻塞 4：正式协议与原型协议还未完全打通
即便第一轮原型已有摘要，后面仍需要把 `Deeper RF` 与传统主链纳入统一比较。

### 阻塞 5：字段语义虽强，但类别值解释可能还不够
`adult` 的优势在于字段有人类语义，但这也意味着：
- 如果 `feature_intro` 太简略；
- 如果类别值映射不清楚；
- 如果 `target_intro` 没把“收入 >50K / <=50K”写明白；

那么 online 反而可能被“看起来很合理”的语义带偏。

---

## 6. 第一轮完成标准

满足下面 6 条，就算 `adult` 完成第一轮接入：
- [x] 数据目录完整
- [ ] DeLTa 传统底座可跑
- [x] `without_llm` 可跑
- [x] `mock_llm` 可跑
- [x] `online_llm` 至少正式尝试并落盘
- [x] 有 `adult` 的统一同口径摘要

---

## 7. 完成后立即要做什么

一旦 `adult` 第一轮接入完成，下一步不要立刻跳数据集，而是先回答 3 个问题：
- `online_llm` 有没有超过 `mock_llm`？
- `online_llm` 有没有超过 `without_llm`？
- `online_llm` 有没有接近或超过 `Deeper RF`？

如果这三个问题还没答清，就不要急着切去 `credit-g`。

---

## 8. 和 `credit-g` 的关系

`credit-g` 是下一张同级别作战卡的候选。

但顺序上建议：
- 先把 `adult` 做成第一份完整的“强语义表格数据接入样板”；
- 再把 `credit-g` 补成和 `adult` 同级别的接入前检查清单与作战卡。
