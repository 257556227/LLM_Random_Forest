# credit-g 数据集接入前检查清单

## 1. 这份清单解决什么问题

这份文档专门回答：

> 如果在 `adult` 之后继续开 `credit-g`，有哪些基础已经具备，哪些缺口必须先补，第一轮应该怎么稳妥推进。

它的定位和 `adult` 的预检清单是同级的，目标是避免“下一站数据集又从头猜一遍”。

---

## 2. 当前核查结论（2026-03-11）

### 2.1 已具备
- `DeLTa-main/dataset_config.py` 已有 `credit-g` 参数配置。
- `DeLTa-main/llm/get_prompts/credit-g.py` 已存在。
- `DeLTa-main/model/llm_rule/credit-g.py` 已存在。
- `DeLTa-main/run_leaf_expansion_mnist.py` 的 `--dataset` 参数是通用的，可复用到 `credit-g`。
- `DeLTa-main/model/leaf_expansion/` 这一层的叶子筛选、特征摘要、relation prompt 与局部扩展逻辑都可复用。

### 2.2 当前缺失
- `DeLTa-main/example_datasets/credit-g/` 已真实生成。
- `DeLTa-main/results/credit-g/` 已有首轮 leaf expansion 结果产物。
- `credit-g` 的同口径摘要脚本与结果文件已补齐。
- `credit-g` 的实验台账与模块映射文档已补齐。

### 2.3 需要特别注意
- 和 `adult` 一样，原型入口默认输出路径仍偏向 `results/mnist/...`，所以真正跑 `credit-g` 时必须显式覆盖 `--output` 和 `--guided_output`。
- `credit-g` 的字段虽然具备金融语义，但如果 `feature_intro` 与类别值说明太弱，LLM 很容易给出“听起来像风险评估”的排序，而不一定是叶子局部最优排序。

---

## 3. 开工前必须完成的检查

### 3.1 数据层检查
- [x] 已确认存在 `DeLTa-main/example_datasets/credit-g/`
- [x] 已准备以下文件：
  - [x] `N_train.npy / N_val.npy / N_test.npy`
  - [x] `C_train.npy / C_val.npy / C_test.npy`
  - [x] `y_train.npy / y_val.npy / y_test.npy`
  - [x] `info.json`
- [x] 已确认 `info.json` 至少包含：
  - [x] `task_type`
  - [x] `num_classes`
  - [x] `n_num_features`
  - [x] `n_cat_features`
  - [x] `task_intro`
  - [x] `feature_intro`
  - [x] `target_intro`
  - [x] `source`

### 3.2 传统 DeLTa 底座检查
- [x] `DELTA_DATASETS=credit-g` 已跑通 `run_randforest.py`
- [ ] `run_get_prompt.py` 能生成 `credit-g` prompt
- [ ] `run_get_answer.py` 能落出 `credit-g` answers（若在线环境不可用，要记录阻塞）
- [ ] `get_trees.py` 能把答案转成规则
- [x] `run.py` 已基于现有 `model/llm_rule/credit-g.py` 跑通训练
- [x] `run_ensemble.py` 已融合并产生日志

### 3.3 Leaf Expansion 原型检查
- [x] `python run_leaf_expansion_mnist.py --dataset credit-g --mode without_llm ...` 已跑通
- [x] `python run_leaf_expansion_mnist.py --dataset credit-g --mode llm_guided --relation_source mock_llm ...` 已跑通
- [x] `python run_leaf_expansion_mnist.py --dataset credit-g --mode llm_guided --relation_source online_llm ...` 已正式尝试并落盘
- [x] 已明确覆盖输出路径：
  - [x] `--output results/credit-g/...`
  - [x] `--guided_output results/credit-g/...`

### 3.4 文档与产物检查
- [x] 已新增 `experiments/credit_g_experiment_ledger.md`
- [x] 已新增 `DeLTa-main/docs/module_map_credit_g.md`
- [ ] 在 `Update.md` 中记录第一轮尝试
- [ ] 把 `credit-g` 结果补进完全通用化文档

---

## 4. 第一轮推荐执行顺序

### 第一步：先补数据层
这一步已完成，`example_datasets/credit-g/` 数据目录已落地。

### 第二步：先跑传统底座
先确认 `credit-g` 在 DeLTa 原链条上能不能稳定产出规则和融合日志。

### 第三步：先跑三组 leaf expansion 原型
目的：
- 确认候选叶子能否稳定选出；
- 确认字段语义和类别值解释是否已经足够支撑 prompt；
- 确认输出路径与结果目录没有继续落到 `mnist`。

### 第四步：当前首轮结论
- `prototype_baseline = 0.7286`
- `without_llm = 0.7171`
- `mock_llm = 0.7286`
- `online_llm = 0.7171`

这说明：
- `mock_llm` 至少能把 `without_llm` 的下滑拉回基线；
- 当前 `online_llm` 还没有在 `credit-g` 上体现出超过 `mock_llm` 的收益。

### 第五步：最后补 `Deeper RF`
把主对比协议补齐成：
- `Deeper RF`
- `without_llm`
- `online_llm`

### 第六步：正式协议最新结论（2026-03-11）
- `deeper_rf = 0.7314`
- `traditional_rf = 0.7429`
- `traditional_delta = 0.7343`
- `online_llm = 0.7171`

这说明：
- `credit-g` 当前已经完成正式主对比协议；
- 但 `online_llm` 仍低于 `Deeper RF`、传统 `RF` 与传统 `DeLTa`；
- 因此下一步重点已经从“补主链基线”切换到“排查为什么金融语义 prompt 没有带来局部排序增益”。

---

## 5. 当前最可能遇到的阻塞

### 阻塞 1：数据目录不存在
这是当前最直接的缺口。

### 阻塞 2：金融字段语义太泛，LLM 容易被“合理故事”带偏
`credit-g` 很适合做语义测试，但也很容易让在线模型偏好“听起来像风险信号”的字段，而不是局部最优分裂字段。

### 阻塞 3：同口径评估脚本缺失
没有 `credit-g` 的统一摘要，后续就很难和 `adult`、`bank` 做横向比较。

### 阻塞 4：结果命名混乱
`credit-g` 文件名本身带连字符，实验台账与产物命名时要尽早统一，否则后续脚本和文档会更容易漂移。

---

## 6. 第一轮完成标准

满足下面 6 条，就算 `credit-g` 完成第一轮接入：
- [x] 数据目录完整
- [x] DeLTa 传统底座可跑
- [x] `without_llm` 可跑
- [x] `mock_llm` 可跑
- [x] `online_llm` 至少正式尝试并落盘
- [x] 有 `credit-g` 的统一同口径摘要

## 8. 当前正式协议判断

- `credit-g` 已经不再停留在“预检 + 首轮原型”，而是正式进入主对比协议阶段；
- 当前结论是：`mock_llm` 能保住基线，但 `online_llm` 还没有转化为正式主对照优势；
- 因此它是当前强语义表格里的反例样板，适合和 `adult` 的正向样板一起做对照分析。

---

## 7. 和 `adult` 的区别

`adult` 更偏“强语义的人类社会字段”；
`credit-g` 更偏“金融风险判断字段”。

因此 `credit-g` 最值得回答的问题是：
- online 的优势到底来自通用自然语言语义；
- 还是来自更精确的局部统计摘要；
- 还是两者必须同时存在才会超过 `mock_llm`。