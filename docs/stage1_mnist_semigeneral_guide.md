# 阶段一：MNIST 半通用化操作说明

## 1. 这份文档解决什么问题

这份文档不是讲抽象方法，而是回答下面几个最实际的问题：
- 现在第一阶段到底先做什么；
- 如果要把 `MNIST` 做成“半通用化可迁移版本”，应该先改哪些地方；
- 如果后面换到别的数据集，优先改哪些文件；
- 调参数时先碰哪些参数，哪些参数先不要乱动。

当前阶段的目标不是直接做完正式版 `LLM-Guided Leaf Expansion Forest`，而是先把现有 `MNIST` 流程整理成一个“换数据时知道去哪改”的稳定底座。

---

## 2. 当前阶段的真实目标

“MNIST 半通用化”在本项目里指的是：
- 保留已经跑通的 `MNIST` 全链路；
- 把数据入口、Prompt 入口、规则入口、训练入口、实验记录入口说明清楚；
- 让后续迁移到新数据集时，不需要重新猜“应该改哪里”；
- 先把工程入口标准化，再做真正的 `relation / leaf expansion` 新方法。

一句话理解：

> 先把 `MNIST` 变成一个可以照着迁移的样板，而不是只跑通一次的实验残留。

### 2.1 为什么现在还要做 `bank` 验证，而不是立刻做 `MNIST` 版 `LLM-Guided Leaf Expansion Forest`

这个问题非常关键，因为它决定了后面工作顺序会不会混乱。

先说结论：

> `MNIST` 半通用化、`bank` 迁移验证、`MNIST` 上的 `LLM-Guided Leaf Expansion Forest` 原型，三者不是同一个任务层次。

更准确的关系是：
- `MNIST` 半通用化：是在整理底座，目标是把当前可跑链路变成稳定样板；
- `bank` 迁移验证：是在验证这个底座是不是真的“可迁移”，而不是只对 `MNIST` 特判有效；
- `MNIST` 上的 `LLM-Guided Leaf Expansion Forest`：是在稳定底座之上，开始真正改树生成机制的原型方法实验。

也就是说，`bank` 的存在不是为了替代 `MNIST`，而是为了回答一个更底层的问题：

> 你现在整理出来的这套入口、命名、规则落地和训练流程，到底是不是“半通用化”的，而不是只对 `MNIST` 这一个样板临时成立。

如果跳过 `bank`，直接在 `MNIST` 上硬做 `LLM-Guided Leaf Expansion Forest`，会有两个风险：
- 一旦结果不好，很难判断是“方法本身没带来增益”，还是“工程底座其实还不稳”；
- 一旦后面要迁到 tabular 数据集，又会重新暴露数据入口、规则命名、结果产物路径这些老问题。

所以当前更合理的顺序不是“放弃 `MNIST`，先去做 `bank`”，而是：

1. 先把 `MNIST` 作为样板整理清楚；
2. 再用 `bank` 验证这套样板是否可迁移；
3. 底座稳定后，再在 `MNIST` 上做第一版 `LLM-Guided Leaf Expansion Forest` 原型；
4. 最后再决定要不要把这个新方法迁到 `bank` 或更多数据集上。

你可以把它理解成两条线：
- 线 A：底座线（`MNIST` 半通用化 -> `bank` 迁移验证）
- 线 B：方法线（`MNIST` 上做 `LLM-Guided Leaf Expansion Forest` 原型）

其中线 B 不能完全脱离线 A，但也不需要等所有迁移都做完才开始。比较稳妥的做法是：
- 先把线 A 收到“入口清楚、产物清楚、迁移清楚”；
- 然后在 `MNIST` 上先起第一版方法原型；
- 再决定是否把新方法扩到 `bank`。

### 2.2 从顶级程序员视角回看：当前有没有偏离原计划

如果严格对照最初的稳妥顺序，当前执行里确实出现过两次“受控偏移”，但它们不是方向跑偏，而是工程节奏上的提前量：

1. `MNIST` 原型线开工，早于 `bank` 真正完成第一轮真实迁移验证。
  - 原始稳妥顺序是：先把 `bank` 迁移验证做完，再正式开 `MNIST` 方法原型。
  - 实际执行里为了避免一直停在文档层，我们先把 `MNIST` 上的 `Leaf Expansion without LLM -> mock_llm/json -> online_llm 入口` 提前搭起来了。

2. `online_llm` 接口接入，早于统一四组协议完全收口。
  - 原始稳妥顺序是：先把 `Baseline RF / Deeper RF / without_llm / llm_guided` 四组协议写齐，再补正式在线 relation。
  - 实际执行里为了先锁定外部接口，我们提前把 `online_llm` 的 provider、参数和摘要结构接通了。

这两次偏移为什么目前仍然可接受：
- 它们没有改变项目主方向，仍然围绕“底座线 + 方法线”推进；
- 提前做出来的原型入口、接口位、测试和文档，后面不会推翻，反而能被后续直接复用；
- `bank` 首轮真实迁移验证后来已经补做完成，所以并没有真的把底座验证永久跳过。

但也要明确说清楚：
- 文档层面的“可以正式开方法原型”条件，在一段时间内曾经落后于真实执行顺序；
- 现在需要把这种偏移写清楚，不然后来的人会以为所有步骤都完全按原计划顺序发生过。

一句话总结：

> 当前不是“战略跑偏”，而是发生过“工程顺序上的受控前插”；现在文档需要把这个事实说透，而不是假装没有发生。

### 2.3 从全局看，阶段一现在到底完成到哪一步

如果站在顶级程序员视角，不看单点改动，而看整个阶段是否收口，可以把当前状态分成三层：

第一层：已经基本坐实的部分
- `MNIST` 已经不再只是一次性实验，而是有主链、参数口径、日志、台账、模块图、测试的样板底座；
- `bank` 已完成第一轮真实迁移验证，证明当前底座并不是只对 `MNIST` 临时成立；
- `MNIST` 新方法原型已经不是纸面概念，而是有 `without_llm`、`mock_llm`、`json`、`online_llm` 入口的真实代码层；
- 文档、日志、看板、测试已经形成基本闭环。

第二层：已经开始，但还没完全收口的部分
- `online_llm` 入口已经接通，真实在线摘要结果也已经成功留档，并在 richer prompt + strict validation + top-k 试分裂后略高于 `mock_llm`；
- `bank` 当前验证的是“本地规则替代在线答案”的闭环，不是完整在线答案闭环；
- `MNIST` 原型目前还是“可运行原型”，还不是“统一四组协议下可重复对照”的正式实验底座。

第三层：当前阶段还明确没做完的部分
- “第一次迁移可以开始 / 还不能开始”的前置条件表已经补齐；
- “底座线 / 方法线在执行层面如何并行而不互相打断”的操作表已经补齐；
- `Baseline RF / Deeper RF / without_llm / llm_guided` 已经有统一库存摘要、统一 runner，且全四组库存阶段都已经真实执行过一轮；
- 当前还额外补出了一份统一 `test_accuracy` 口径的同口径摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`；
- `bank` 的在线 `run_get_answer.py -> get_trees.py` 仍然没有恢复。

所以更准确的阶段判断不是“阶段一已经完全结束”，而是：

> 阶段一的底座已经从“抽象说明”进入“真实可迁移”，但距离“彻底收口并可无歧义交接”还差最后一轮协议化和阻塞收口。

---

## 3. 当前最重要的几个板块

### 3.1 数据板块：换数据最先看这里
- 数据目录：`DeLTa-main/example_datasets/<dataset>/`
- 数据配置：`DeLTa-main/dataset_config.py`
- 默认/搜索配置：
  - `DeLTa-main/configs/default/DeLTa.json`
  - `DeLTa-main/configs/default/RandomForest_save.json`
  - `DeLTa-main/configs/classical_configs.json`

你要做的数据相关动作主要是：
- 确保数据目录里有 `N_train/N_val/N_test` 与 `y_train/y_val/y_test`；
- 在 `dataset_config.py` 里把新数据集名字接进去；
- 检查这个数据集是分类还是回归、类别特征有没有单独处理要求；
- 再决定是否需要改默认配置文件。

### 3.2 基线树板块：先保证普通 RF 稳定
- 入口：`DeLTa-main/run_randforest.py`
- 训练主逻辑：`DeLTa-main/train_model_classical.py`
- 当前树模型实现：`DeLTa-main/model/randomforest_save.py`

如果你换数据后第一步就报错，优先查这三处。

因为后面的 Prompt、LLM、融合，都是建立在“先成功产出基础规则”之上的。

### 3.3 Prompt 板块：换数据时一定要补模板
- Prompt 入口：`DeLTa-main/llm/get_prompts/run_get_prompt.py`
- 当前 `MNIST` 模板：
  - `DeLTa-main/llm/get_prompts/mnist.py`
  - `DeLTa-main/llm/get_prompts/mnist_stable.py`
  - `DeLTa-main/llm/get_prompts/mnist_ab_b.py`
  - `DeLTa-main/llm/get_prompts/mnist_prompt12_aligned.py`

如果你换数据集，最少要做一件事：
- 新增同名 Prompt 文件，例如 `<dataset>.py`。

如果你只是做 A/B/C 对照：
- 不要动训练参数；
- 只替换 Prompt 模板；
- 用 `DELTA_PROMPT_TEMPLATE` 控制模板切换。

### 3.4 在线答案与规则落地板块：LLM 输出能不能真正接上训练，看这里
- 查询入口：`DeLTa-main/llm/query/run_get_answer.py`
- 查询实现：`DeLTa-main/llm/query/get_answer.py`
- 规则解析：`DeLTa-main/llm/get_trees.py`
- 规则落地目录：`DeLTa-main/model/llm_rule/`

这里的核心不是“问出来”，而是“问出来的结果能不能被稳定解析并继续训练”。

换数据时重点关注：
- 规则输出格式是否仍能被 `get_trees.py` 接住；
- `model/llm_rule/<dataset>.py` 是否生成成功；
- 规则命名是否和训练入口预期一致。

### 3.5 训练与融合板块：结果最终有没有提升，看这里
- 训练入口：`DeLTa-main/run.py`
- 融合入口：`DeLTa-main/run_ensemble.py`
- 融合实现：`DeLTa-main/ensemble.py`
- 常用工具：`DeLTa-main/model/utils.py`

这里重点关注：
- `--num_answers`：当前要训练多少份答案；
- `--n_ensemble`：当前融合多少路结果；
- `--eta`：融合权重；
- 最终日志是否还能稳定产出 Accuracy / AUC。

### 3.6 文档与实验台账板块：不然很容易再次丢上下文
- 进度清单：`lookme.md`
- 正式日志：`Update.md`
- 已知边界：`docs/project_known_info.md`
- 本文档：`docs/stage1_mnist_semigeneral_guide.md`
- 模块映射：`DeLTa-main/docs/module_map_mnist.md`
- 实验台账：`experiments/mnist_experiment_ledger.md`

规则很简单：
- 做完一轮改动，先写 `Update.md`；
- 确认阶段状态变化，再改 `lookme.md`；
- 如果是长期结论，再写进 `docs/project_known_info.md` 或实验台账。

### 3.7 五类入口速查表（入口 -> 常见产物 -> 排查位置）

| 入口类型 | 先看哪里 | 常见产物 | 出问题先查哪里 |
| --- | --- | --- | --- |
| 数据入口 | `DeLTa-main/example_datasets/mnist/`、`DeLTa-main/dataset_config.py` | `N_train/N_val/N_test.npy`、`y_train/y_val/y_test.npy`、`info.json` | 数据文件是否齐全、`dataset_config.py` 是否接入、任务类型是否一致 |
| 基线树入口 | `DeLTa-main/run_randforest.py`、`DeLTa-main/train_model_classical.py`、`DeLTa-main/model/randomforest_save.py` | `DeLTa-main/results/mnist/mnist_md8_ml20_tree3.log`、`mnist_md8_ml20_tree3.npy` | 树深/叶子/树数口径、保存路径、训练是否正常结束 |
| Prompt/查询入口 | `DeLTa-main/llm/get_prompts/run_get_prompt.py`、`DeLTa-main/llm/query/run_get_answer.py` | `DeLTa-main/llm/prompts/mnist/*.txt`、`DeLTa-main/llm/answers/mnist/*.txt` | Prompt 模板是否切对、`OPENAI_*` 环境变量、限流/分片参数 |
| 规则与训练入口 | `DeLTa-main/llm/get_trees.py`、`DeLTa-main/model/llm_rule/mnist.py`、`DeLTa-main/run.py`、`DeLTa-main/run_ensemble.py` | `RF_md*_full_cart_*.npy`、`e_RF_*.log` | 规则名是否和训练入口一致、答案数量、融合索引与 `eta` |
| 原型入口 | `DeLTa-main/run_leaf_expansion_mnist.py`、`DeLTa-main/model/leaf_expansion/` | `leaf_expansion_without_llm_summary.json`、`llm_guided_leaf_expansion_summary.json`、`llm_guided_leaf_expansion_json_summary.json` | 候选叶子筛选、relation 来源、json relation 文件路径、`OPENAI_*` 环境变量、`--llm_model/--llm_max_retries/--llm_retry_delay/--llm_request_interval`、局部扩展是否写回 |

---

## 4. 如果后面要换数据集，建议按这个顺序改

### 第一步：复制 `MNIST` 的数据组织方式
先保证新数据集也具备：
- `N_train.npy / N_val.npy / N_test.npy`
- `y_train.npy / y_val.npy / y_test.npy`
- `info.json`

### 第二步：在 `dataset_config.py` 注册数据集
如果这一步没做，后面的脚本虽然能跑，但通常会在配置读取、路径拼接或任务类型判断时出问题。

### 第三步：新增对应 Prompt 模板
最少先补：
- `DeLTa-main/llm/get_prompts/<dataset>.py`

如果是正式对照实验，再决定要不要补：
- `<dataset>_stable.py`
- `<dataset>_ab_b.py`
- 其他实验模板

### 第四步：先只跑 baseline，不要一上来跑全链
推荐顺序：
1. `run_randforest.py`
2. `run_get_prompt.py`
3. `run_get_answer.py`
4. `get_trees.py`
5. `run.py`
6. `run_ensemble.py`

任何一步挂掉，都先回到上一层，不要一口气同时改多处。

### 第五步：确认规则命名与结果命名一致
历史上 `MNIST` 已经踩过一个坑：
- 本地规则文件命名与训练入口预期命名不一致。

所以换新数据时，要额外检查：
- `model/llm_rule/<dataset>.py` 里导出的规则名；
- `run.py` / `run_ensemble.py` 实际读取的结果名；
- 结果目录下 `.npy` 与 `.log` 是否按同一口径落盘。

---

## 5. 参数应该先调哪里

### 5.1 第一优先：先保证口径稳定
优先固定这些：
- 数据集
- 随机种子
- 基线树深度 / 树数
- 训练答案数量
- 融合数量

### 5.2 当前最常碰的参数入口
- 树基线相关：
  - `md` / `ml` / `tree` 对应的深度、叶子和树数量口径
- 查询相关：
  - `--num_queries`
  - `--request_interval`
  - `--max_completion_tokens`
  - `--normalize_tree_output`
- 训练/融合相关：
  - `--num_answers`
  - `--n_ensemble`
  - `--eta`

### 5.3 推荐调参顺序
1. 先调在线答案数量是否足够稳定；
2. 再调融合数量；
3. 最后再调 `eta`；
4. Prompt 对照放在这些都稳定以后。

不建议一开始就同时改：
- Prompt
- 树深度
- 查询数量
- 融合权重

不然最后很难知道到底是哪一项起作用。

---

## 6. 第一阶段 TODO（执行拆解版）

这一节不是泛泛而谈，而是当前阶段真正要照着推进的拆解清单。

### 6.1 当前总目标

当前阶段只做一件事：

> 把 `MNIST` 整理成一个“能稳定复用、能指导第一次迁移、能减少重复踩坑”的样板工程。

所以这里的 TODO 不是“做更多实验”，而是“先把入口、产物、约束和验收收紧”。

### 6.2 顶级程序员视角的任务拆解

#### 任务 A：固定样板口径
- [x] 固定 `MNIST` 作为阶段一样板数据，不再同时扩别的数据集。
- [x] 固定当前稳定入口：`run_randforest.py -> run_get_prompt.py -> run_get_answer.py -> get_trees.py -> run.py -> run_ensemble.py`。
- [x] 固定当前常用参数口径：`md8/ml20/tree3`、`--num_answers`、`--n_ensemble`、`--eta`。

验收标准：
- 任何人接手时都知道先跑哪条链，不会出现“这轮到底用哪个脚本组合”的歧义。

#### 任务 B：固化五类入口
- [x] 把“数据入口 / Prompt入口 / 规则入口 / 训练入口 / 实验台账入口”文档化并保持同步。
- [x] 把“换数据时去哪改”的稳定清单写进当前文档。
- [x] 把每类入口的典型产物路径补成一张一眼能查的速查表。

验收标准：
- 不打开代码，也能知道“我要改数据、改 Prompt、查规则、查结果”分别去哪里。

#### 任务 C：补齐收口标准
- [x] 给 `MNIST` 半通用化补一个“完成 / 未完成”的硬验收定义。
- [x] 给第一次迁移补一个“可以开始 / 还不能开始”的前置条件表。
- [x] 把当前阶段的阻塞项和非阻塞项分开写清楚。

验收标准：
- 后续讨论时，不再用“差不多可以了”这种模糊判断，而是直接对照清单判断状态。

#### 任务 D：把迁移准备和新方法开发解耦
- [x] 明确“半通用化底座”和“`relation / leaf_expansion` 方法实验”不是同一件事。
- [x] 在执行层面把“底座稳定工作”与“新方法探索工作”拆成两条并行但不互相打断的线。

验收标准：
- 后续即使新方法表现不好，也不会把底座问题和方法问题混在一起。

#### 任务 E：完成第一次非图像迁移前置准备
- [x] 选定第一个非图像迁移对象为 `bank`。
- [x] 单独写出 `bank` 的迁移说明。
- [x] 按迁移说明完成一次真实链路验证并留存产物。

验收标准：
- `bank` 至少完成一轮从基线到结果日志的真实执行，且卡点能被准确记录。

### 6.2.1 第一次迁移“可以开始 / 还不能开始”前置条件表

| 检查项 | 可以开始的状态 | 还不能开始的信号 |
| --- | --- | --- |
| 数据入口 | `example_datasets/<dataset>/` 已具备 `N_*`、`y_*` 与 `info.json` | 数据文件缺失，或训练/验证/测试划分不完整 |
| 配置入口 | `dataset_config.py` 已注册数据集，任务类型已确认 | 运行脚本只能识别旧数据集，或分类/回归口径不清楚 |
| Prompt 入口 | `llm/get_prompts/<dataset>.py` 已存在，能被 `run_get_prompt.py` 找到 | 仍需要手改脚本路径，或模板文件缺失 |
| 规则落地 | `model/llm_rule/<dataset>.py` 能稳定生成或已有本地规则替代方案 | Prompt 已生成，但规则无法解析或规则名对不上训练入口 |
| 训练与融合 | 至少能跑通一轮 baseline -> train -> ensemble 并留日志 | 训练产物、融合日志、规则命名三者对不上 |
| 记录与回归 | `Update.md`、`lookme.md`、实验台账、测试已同步 | 真实动作已经发生，但文档和测试没有追平 |

判断原则很简单：
- 只要“数据 / 配置 / Prompt / 规则 / 训练 / 记录”六项里还有两项以上不稳定，就不应该宣称“可以开始第一次迁移”；
- 如果只剩在线答案链被外部网络阻塞，但本地规则替代链已经能闭环，那么可以把它定义为“底座可迁移，在线链待恢复”。

### 6.2.2 底座线 / 方法线并行执行表

| 线别 | 当前目标 | 允许并行做什么 | 不能跨线混淆什么 | 当前代表产物 |
| --- | --- | --- | --- | --- |
| 底座线 | 保证入口稳定、命名稳定、迁移可复用 | 修路径、修依赖、补文档、补回归测试、做 `bank` 迁移闭环 | 不能把方法收益问题误判成入口/依赖问题 | `docs/stage1_mnist_semigeneral_guide.md`、`docs/stage1_bank_migration_guide.md`、`experiments/bank_experiment_ledger.md` |
| 方法线 | 验证 `LLM-Guided Leaf Expansion` 有没有真实信号 | 做 `without_llm`、`mock_llm`、`json`、`online_llm` 原型，补协议摘要 | 不能把原型单次结果直接当成最终同口径结论 | `DeLTa-main/run_leaf_expansion_mnist.py`、`docs/mnist_llm_guided_leaf_expansion_prototype.md`、`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json` |

执行原则：
- 底座线优先解决“能不能稳定跑、能不能迁移、能不能交接”；
- 方法线优先解决“有没有信号、增益来自哪里、失败卡在哪层”；
- 两条线可以并行，但任何一条线的结论都不能拿去替代另一条线的验收。

### 6.3 现在就该做的动作顺序

如果今天就开始推进，建议只按下面顺序做，不要跳：

1. 先把 `MNIST` 样板文档补到“入口明确 + TODO明确 + 验收明确”；
2. 再检查 `MNIST` 当前产物是否仍完整、命名是否一致；
3. 先把五类入口的速查表补齐；
4. 然后按 `docs/stage1_bank_migration_guide.md` 开始第一次 `bank` 迁移；
5. 迁移过程中只修当前层，不顺手大改；
6. `bank` 迁移达到“能判断底座是否可迁移”的程度后，再开 `MNIST` 上的 `LLM-Guided Leaf Expansion Forest` 原型；
7. 完成后把结果、阻塞和下一步收口动作写回 `Update.md`。

上面这 7 步是原始稳妥顺序，它的价值是帮你理解“阶段边界为什么这么划”。

但由于当前真实执行已经往前推进过一截，今天再继续时，建议改用下面这套“按现状收口”的动作顺序：

1. 先补齐“第一次迁移可以开始 / 还不能开始”的前置条件表；
2. 再把“底座线 / 方法线并行但不互相打断”的执行表写清楚；
3. 先生成四组协议的统一库存摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`；
4. 生成四组协议的统一 `test_accuracy` 同口径摘要；
5. 再尝试补一轮真实 `online_llm` 摘要结果，并尽量纳入同口径摘要；
6. 如果 `online_llm` 仍被网络阻塞，就单独把这件事记为“外部依赖阻塞”，不要继续污染主协议；
7. 最后再决定是否把 `LLM-Guided Leaf Expansion` 从 `MNIST` 迁到 `bank`。

### 6.4 本周交付物清单

当前阶段至少应该沉淀出这些交付物：
- `docs/stage1_mnist_semigeneral_guide.md`：阶段一样板操作说明；
- `docs/stage1_bank_migration_guide.md`：第一次非图像迁移说明；
- `lookme.md`：当前阶段收口状态；
- `Update.md`：每一轮真实动作、阻塞和结论；
- `test/test_project_docs.py`：文档结构与入口引用回归。

### 6.5 当前未完成项

- [x] 给五类入口补一张“入口 -> 常见产物 -> 排查位置”的速查表。
- [x] 给 `MNIST` 半通用化补一节“完成标准 / 阻塞标准”。
- [x] 给 `bank` 迁移补一轮真实链路验证结果。
- [x] 已单独开出 `MNIST` 上的第一版 `LLM-Guided Leaf Expansion Forest` 原型任务说明：`docs/mnist_llm_guided_leaf_expansion_prototype.md`。
- [x] 已新增 `MNIST` 原型实验入口 `DeLTa-main/run_leaf_expansion_mnist.py`，并完成首轮 `Leaf Expansion without LLM` 对照运行。
- [x] 已在当前原型入口上接入最小版 `LLM relation`（`mock_llm` / `json`），并完成第一版 `LLM-Guided Leaf Expansion` 对照运行。
- [x] 已在当前原型入口上接入正式在线 `LLM relation` 入口（`online_llm`），支持通过 `OPENAI_*` 与 `--llm_*` 参数控制在线调用。
- [x] 已补齐五类入口速查表，可直接对照入口、常见产物和排查位置。
- [x] 已新增 `bank` 对应的实验台账与模块映射文档：`experiments/bank_experiment_ledger.md`、`DeLTa-main/docs/module_map_bank.md`。
- [x] 已新增四组协议统一库存摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`，并补充生成脚本 `DeLTa-main/tools/build_mnist_protocol_summary.py`。
- [x] 已新增四组协议统一 runner 与配置清单：`DeLTa-main/tools/run_mnist_four_way_protocol.py`、`DeLTa-main/configs/leaf_expansion/mnist_four_way_protocol.json`。
- [x] 已用统一 runner 真实执行四组库存协议的全部阶段（`baseline_rf / deeper_rf / without_llm / llm_guided / build_summary`）。
- [x] 已生成统一同口径摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`，当前 `metric_scope = same_test_accuracy_eval`。
- [x] 已把真实 `online_llm` 的正式尝试状态挂入同口径摘要：`online_llm_attempt`。
- [x] 已拿到 richer prompt + strict validation + top-k 试分裂后的真实 `online_llm` 成功精度结果：`test_accuracy = 0.5622`，并已纳入同口径摘要。
- [x] 已补 `mnist_leaf_expansion_oracle_analysis.json`，用于判断 `leaf 8/9` 的单特征上界与当前 gap。
- [x] 已把 `MNIST` online relation 重复实验补到 5 次，并确认 `online_llm_minus_mock_llm = +0.0019` 在当前 5/5 次中稳定。
- [x] 已完成一轮轻量 `bank` 专用 prompt 增强验证：去掉 tabular 数据上的伪 `pixel` 语义，并补入表格特征指引。

### 6.1 当前推荐的第一个非图像迁移对象

当前更推荐优先选 `bank`，不是 `house_16H_reg`。

原因：
- `bank` 已有完整数据目录；
- `bank` 是分类任务，更接近当前已跑稳的 `MNIST` 主链；
- `bank` 已经有 `dataset_config.py` 参数与 Prompt 模板；
- 第一轮更适合先做“分类迁移验证”，而不是直接切到回归 + `tabpfn`。

对应执行说明见：`docs/stage1_bank_migration_guide.md`。

---

## 7. 当前完成标准与阻塞标准

### 7.1 什么叫“MNIST 半通用化已完成”

满足下面 5 条，才算这一阶段真正完成：
- `MNIST` 主链入口、主要参数入口、主要产物入口都写清楚；
- 当前样板文档能指导第一次非图像迁移，不需要重新口头解释；
- `lookme.md`、`Update.md`、本文档三者状态一致；
- 文档回归测试能覆盖关键入口与引用关系；
- 至少有 1 个非图像数据集开始按这套说明执行。

### 7.2 什么叫“还处于阻塞中”

出现下面任一情况，都说明阶段还没收口：
- 文档知道做什么，但一上手仍不知道先改哪个文件；
- `MNIST` 现有产物命名和脚本读取口径不一致；
- 迁移时一出错就需要同时翻多个入口；
- 新增数据集后无法判断是数据问题、规则问题还是训练问题；
- 新方法讨论开始挤占底座收口工作，导致阶段目标漂移。

### 7.3 什么叫“可以开始 `MNIST` 版 `LLM-Guided Leaf Expansion Forest` 原型”

满足下面 4 条，就可以正式开方法原型，而不是继续只改文档：
- `MNIST` 的主链入口、参数入口、产物命名已经稳定；
- 五类入口速查表已经补齐；
- `bank` 迁移已经至少完成一轮，能说明底座是否具备基本可迁移性；
- 当前团队已经统一：接下来是在做“方法实验”，不是继续做底座修补。

正式原型说明见：`docs/mnist_llm_guided_leaf_expansion_prototype.md`。

当前已完成首轮无 LLM 原型运行，摘要结果见：`DeLTa-main/results/mnist/leaf_expansion_without_llm_summary.json`。

当前也已完成首轮最小版 `LLM-Guided Leaf Expansion` 运行，摘要结果见：`DeLTa-main/results/mnist/llm_guided_leaf_expansion_summary.json`。

当前已完成 `bank` 第一次真实迁移验证（使用本地规则文件替代在线答案）：
- 基线日志：`DeLTa-main/results/bank/bank_md20_ml50_tree15.log`
- Prompt 产物：`DeLTa-main/llm/prompts/bank/bank_randfull_md20_ml50_tree15.txt`
- 训练产物：`DeLTa-main/results/bank/RF_md20_ml50_tree15_full_cart_0.npy`
- 融合日志：`DeLTa-main/results/bank/e_RF_md20_ml50_tree15_bank_cart_0.log`
- 当前结果：`RF Accuracy MEAN = 0.8986`，`Fused Accuracy MEAN = 0.8991`

当前还已补上正式在线 `LLM relation` 入口：
- `--relation_source online_llm`
- `--llm_model`
- `--llm_max_retries`
- `--llm_retry_delay`
- `--llm_request_interval`

注意：这一步当前表示“接口已接通”，是否产出真实在线对照摘要仍取决于运行环境里是否提供可用的 `OPENAI_*` 凭据与网络。

当前最新正式尝试结果（2026-03-08）：
- 尝试文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json`
- 当前状态：`status = success`
- 当前结果：`baseline_accuracy = 0.4953`，`test_accuracy = 0.5622`，`llm_model = gpt-4o`。
- 当前影响：同口径摘要已新增 `experiments.online_llm`，并记录 `online_llm_minus_mock_llm = +0.0019`，说明真实在线 relation 已成功纳入正式对照，并首次略高于 `mock_llm`。
- 当前补充：已新增 `DeLTa-main/results/mnist/mnist_leaf_expansion_oracle_analysis.json`；其中 `leaf 8` 的 oracle 最优单特征是 `347`，当前 `online_llm` 已命中该特征。
- 历史排查：早先确实出现过 `Cloudflare` 拦截与 `401 invalid_api_key`，这些问题现在已经降级为历史排查分支，不再是当前最新状态。

### 7.4 从顶级程序员视角看，当前还需要继续什么

如果只保留最高优先级，当前还需要继续的事情可以压缩成下面 4 件：

1. 先补齐阶段一最后两个文档性缺口
- 这一步已经完成；
- 当前不再缺文档结构，也已经补出一版正式同口径摘要；
- 现在 `MNIST` 上 5 次重复已经确认这次 `online_llm` 的微弱反超稳定；
- 当前更缺的是判断这条增益迁到 `bank` 时为什么会停在“语义更合理，但精度未提升”。

2. 再把 `MNIST` 原型从“能跑”升级成“协议稳定”
- 关键不是继续堆新入口；
- 当前已经有统一库存摘要，也已经有统一 `test_accuracy` 同口径摘要；
- 当前也已经补上一份正式 `online_llm` 成功记录；
- 当前也已经补上一份 `oracle` 叶子级上界分析；
- 当前这一步已经通过 5 次重复完成；下一步更有价值的是解释为什么 `bank` 的轻量语义增强没有继续带来 test accuracy 增益。

3. 再单独处理在线阻塞，而不是把它混进阶段判断
- `online_llm` 当前已经不是“阻塞未通”，而是“已跑通且有结果”；
- `bank` 的在线答案链当前仍可能遇到同类外部依赖问题；
- 所以当前更适合把 `MNIST` 视为已完成一次真实在线验证，再把外部依赖风险单独迁移管理到后续数据集。

4. 最后再决定是不是把新方法扩到 `bank`
- 只有当 `MNIST` 的统一协议先稳定下来，迁到 `bank` 才有解释力；
- 不然会把“方法增益”“数据集差异”“在线阻塞”三件事搅在一起。

### 7.5 当前阶段最推荐的下一步

如果只选一个最值得继续的方向，当前最推荐的是：

> 当前 `MNIST` 的 `+0.0019` 已经在 5 次重复中稳定；下一步更推荐围绕 `bank` 做分层语义增强，先从轻量 tabular prompt 适配入手，再判断是否需要更重的 task/target 定制。

原因很简单：
- `bank` 已经完成了“底座可迁移”的最低验证；
- `online_llm` 接口位也已经接好且成功执行过；
- `mock_llm` 与真实 `online_llm` 的同口径摘要都已经补出；
- 现在最缺的不是“有没有成功调用”，而是“为什么这条已稳定的微弱增益迁到 `bank` 后没有继续保住”；
- 历史阻塞已经被正式收口到 `mnist_leaf_expansion_online_llm_attempt.json` 与同口径摘要里，不再是当前主问题。

---

## 8. 和后续新方法怎么衔接

当前这份文档服务的是“半通用化底座”，不是最终方法本体。

等这一步完成后，后续代码组织建议会自然分成四块：
- `baseline/`：普通 RF 与 deeper RF；
- `relation/`：LLM 提供的 feature relation；
- `leaf_expansion/`：局部叶子扩展；
- `experiments/`：统一实验入口与记录。

也就是说：

> 现在做的是“把地基理顺”，下一步才是“把新方法真正插进去”。
