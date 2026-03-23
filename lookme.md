# LLM-Guided Leaf Expansion Forest - 执行清单

> 注意：本项目已从初期的 DeLTa 代码复现，演进为探索 **LLM-Guided Leaf Expansion Forest** 的新机制实验。以下清单旨在统一这一新机制的实验范式，约束我们在“受控”的边界内做实验，而不是散落地瞎跑。

## 当前项目收敛与推进工作流

其实就是Checklist。

### 1：框架定名
- [x] 统一命名成一个独立的方法，不再混在 DeLTa Prompt 改进里讲。现正式定名：**Relation-Aware Random Forest** 或 **LLM-Guided Leaf Expansion Forest**。
- [x] 明确其核心不仅仅是输出上的 fusion，而是在树模型末端做局部结构扩展，并用 LLM 提供 feature relation 先验。

###  2：统一实验入口
- [x] 用一个统一的实验脚本来替代过去散落的抽取、合成、评估链条。
- [x] 已在 `DeLTa-main/run_leaf_expansion_mnist.py` 实现通过参数无缝切换 `baseline_rf`、`deeper_rf`、`without_llm` 以及 `llm_guided (包含 mock_llm, online_llm)` 等模式。

###  3：统一数据集适配
- [x] 半通用化：不能只测 MNIST 以防过拟合。
- [x] 目前已确保至少 `MNIST`（图像）+ 一份非图像数据集（`bank`）在同一套管道中能跑。

###  4：统一结果落盘
- [x] 评估口径必须同源。统一采用生成的 `*_leaf_expansion_same_metric_summary.json` 文件。
- [x] 系统级自动记录：Accuracy，并包含扩展的叶子数、采用的深度、时间、树的复杂度描述。
- [x] 当前同口径 MNIST `online_llm (0.5622)` 已稳定超越 `mock_llm (0.5603)`。

###  5：关系图 / relation_map 单独模块化
- [x] 为了不让特征先验跟主实验逻辑耦合太深，不能散落在实验调度里。
- [x] 已经重构到 `DeLTa-main/model/leaf_expansion/mnist_relation_provider.py` 与 `mnist_leaf_selector.py`。在这里专门对付提示工程、上下文封装、以及大模型验证。

###  6：失败结果分析
- [x] **失败也有价值**：之前我们的 Bank 数据集由于 Prompt 设计偏差（把 tabular 当成了类似 pixel 看待，缺乏局部约束），导致 `online_llm` 反而跑不过基于 impurity 的 `mock_llm` (0.8886 vs 0.8890)。
- [x] 未将此作为废案抛弃，而是用 `compare_bank_leaf_expansion_online_vs_mock.py` 在叶子粒度排查并得出结论：补足 Counterfactual Check 和局部准确率指引后，挽回了这种误判（让 bank 的 online_llm 最新跑出 0.8890，平齐 mock_llm）。

---

## 历史阶段验收
- [x] 单数据集全链路稳定复现 (原 DeLTa + TnT链路)
- [x] 迁移后结果可与 RF 基线对比 (A/B/C 版本对齐)
- [x] Prompt 优化有量化收益验收
- [x] 产出了流程图与核心模块映射表