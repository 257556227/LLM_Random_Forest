# 项目介绍：LLM-Guided Leaf Expansion Forest

> 本项目已**不再仅仅是 DeLTa 的 Prompt 改进版**，而是一个新的机制探索：**Relation-Aware Random Forest (或 LLM-Guided Leaf Expansion Forest)**。  

## 1. 核心思想（方法边界）

传统模型负责“打底”，大模型负责“提供特征的先验关系”。但是核心**不再是纯粹的全局 Fusion**。  
真正在做的是：

- **在传统树模型（Random Forest）的末端（叶子节点）增加局部结构扩展**。
- **让大语言模型（LLM）通过特征间的物理或业务含义 (Feature Relation Priors)，帮助在容易混淆的高杂质叶子下选择最佳的特征进行展开。**

简而言之，这不是一个简单的“LLM 代替树”或“纯加权平均”的项目，而是一个**深入树模型的生成机制，让 LLM 局部介入结构生长**的方法。

---

## 2. 项目结构

项目结构（在 `DeLTa-main/` 与统一脚本中）清晰地切分为：

- `baseline/`（对应代码逻辑）：普通 RF、deeper RF。
- `relation/`（在 `model/leaf_expansion/` 下）：负责大模型的特征关系解释与提取。
- `leaf_expansion/`（在 `model/leaf_expansion/` 下）：末端扩展生长逻辑（支持 top-k 候选择优）。
- `experiments/` 与 `results/`：固化的实验配置和跑出的同口径结果日志。
- `README`（本文件）：讲项目怎么跑、已知问题、核心认知。

---

## 3. 统一实验协议

现在所有的结果必须是同口径的。在代码里已经强制统一了：
- **数据集**：目前已保障跑通 **MNIST** 以及至少一个非图像数据集（**bank** 营销数据集）。
- **比较基线**：统一树数、统一深度；明确对比“是否扩展”、“是否用 LLM (without_llm / mock_llm / online_llm)”。
  - mock_llm本质上不是一个真实的大语言模型，而是一个人为硬编码的、确定性的“启发式排序规则”，被用作我们实验中的一条测试基线,（位于 mnist_relation_provider.py），mock_llm 并没有做任何推理。它仅仅是把传进来的候选特征列表做了一个基于叶子 ID 的简单轮转（Rotation）：排名的逻辑仅仅是 leaf_id % len(candidate_ids)
- **客观指标**：统一落盘 Accuracy 测试集准确率、扩展涉及的叶子节点数比较（见各种 `_summary.json`）。

---

## 4. 当前的核心结论（包含“失败分析”价值）

**目前已经验证半通用化的 MNIST（能跑版本）：**
- `baseline_rf`、`deeper_rf`、`without_llm` 以及假定有最好先验的 `mock_llm` (0.5603) 皆已作为协议长期跑通。
- 引入了 `richer prompt`、`严格校验` 和 `top-k试分裂择优` 后，真实的 **`online_llm` 在 MNIST 上达到了 0.5622，稳定且略好于 `mock_llm`**。这证明了的“叶子扩展+LLM先验”机制走通了闭环，增益是真实的。

**在非图数据集 (Bank) 上的适配（迁移适配）：**
- 没有只测图像。Bank 数据跑起来了。
- 最开始 online 的表现因为特征理解偏差，导致不敌 mock（失败的探索）；
- 但随后通过引入 Explicit Counterfactual Check（让模型自己反思比如“年龄vs通话时间”的区别）和提供局部特征质量信号（Accuracy），**Bank 的 online_llm 达到了 0.8890，成功追平了 mock_llm**。证明跨模态轻量适配有效。

---

## 5. 一步一步跑

> 默认在 Windows，且已有 conda 环境 `py310`。请在根目录执行验证。

### 第 1 步：查阅代码被改了哪里
打开 `code_modification_mapping.md`。看看现在的核心逻辑脚本在哪，不要再去老旧文件里找。

### 第 2 步：准备环境和配置
除了激活 `py310`，如果你要跑真实大模型，填好 `DeLTa-main/local_openai_config.json`（避免 API 错误）。

### 第 3 步：一键运行统一实验入口（以 MNIST 为例）
现在有统一脚本，不再是以前乱七八糟分离的了。
```bash
cd DeLTa-main
# 示例：跑没有 LLM 的叶子扩展（看基础扩展的下界）
python run_leaf_expansion_mnist.py --mode without_llm --save_summary
```

### 第 4 步：去同口径汇总里看结果
所有的精度指标都会生成并落在 `DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json` 或 `bank` 的目录下。

---

## 6. 常见已知问题

1. **在线查询报 401 / 拦截**
   - 不再是用环境变量猜了，请去修改 `DeLTa-main/local_openai_config.json` 写入有效 key。
2. **凭感觉觉得精度“变好了”**
   - 不看体感，只看统一同口径的 `evaluate_mnist_same_metric_protocol.py` 脚本结算出来的 json 文件。

---

## 接下来怎么继续？（接手必看）

1. **想知道具体TODO**：看本目录下的 `lookme.md`。
2. **想改代码**：先看 `code_modification_mapping.md` 然后再动手。