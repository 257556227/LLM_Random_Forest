# 代码修改映射表 (Code Modification Mapping)

这份文档旨在记录 **LLM-Guided Leaf Expansion Forest** 原型开发过程中，哪些核心代码被新增或修改了，以及如何运行现有代码。

**最后更新日期: 2026年3月23日**

---

## 2026-03-23 新增

| 类别 | 文件路径 | 说明 |
|:---|:---|:---|
| 数据下载 | download_datasets.py | 下载 sklearn 数据集脚本 |
| 数据集 | DeLTa-main/example_datasets/california_housing/ | 刚补全的回归数据集 |
| 优化参数 | --top_k_leaves N | 增加扩展叶子数量 |
| 优化参数 | --base_max_depth N | 调整基线树深度 |

---

## 可复现的优化命令

### bank 最佳配置（超越 deeper_rf）
```bash
cd DeLTa-main
conda activate py310
python run_leaf_expansion_mnist.py --dataset bank --mode llm_guided --relation_source mock_llm --top_k_leaves 5 --save_summary
# 结果：89.04% vs deeper_rf 88.73%
```

### mnist 最佳配置（大幅提升）
```bash
cd DeLTa-main
conda activate py310
python run_leaf_expansion_mnist.py --dataset mnist --mode llm_guided --relation_source mock_llm --base_max_depth 4 --save_summary
# 结果：61.68% vs baseline 49.53%
```

---

## 1. 核心目录与文件映射

| 类别 | 文件路径 | 核心功能与变动说明 |
|:---|:---|:---|
| **实验主入口** | `DeLTa-main/run_leaf_expansion_mnist.py` | **新增**：统一实验入口，一键支持 `baseline_rf`, `deeper_rf`, `without_llm`, `llm_guided` 四种模式及不同数据集的切换 (`MNIST`, `bank`)。|
| **特征关系提取** (Relation) | `DeLTa-main/model/leaf_expansion/mnist_relation_provider.py` | **新增**：把关系提取独立模块化。负责向 LLM 构造包含 `task_intro`、`counterfactual check`、`one_feature_train_accuracy` 等特征描述的 Prompt，并严校验返回结果。 |
| **叶子上下文筛选** | `DeLTa-main/model/leaf_expansion/mnist_leaf_selector.py` | **新增**：负责在 RF 树的末端筛选出“高杂质”叶子，为候选特征计算信息增益、计算单特征准确率，并将这些前置信息准备给关系提取模块。 |
| **末端扩展逻辑** (Expansion) | `DeLTa-main/model/leaf_expansion/mnist_leaf_expander.py` | **新增**：负责在指定的叶子节点下方挂载 stump（单层扩展），现在支持 `top-k` 择优机制而不是无脑吃 top-1。 |
| **统一结果与评估** | `DeLTa-main/tools/evaluate_mnist_same_metric_protocol.py`<br>`DeLTa-main/tools/evaluate_bank_leaf_expansion_same_metric.py` | **新增**：统一 Accuracy 等指标落盘逻辑，保障同口径下的对比不会漂移。 |
| **局部分析工具** | `DeLTa-main/tools/compare_bank_leaf_expansion_online_vs_mock.py` | **新增**：针对 bank 数据集，逐个叶子比对 `online_llm` 和 `mock_llm` 以及局部训练/测试效果，用数据解释差异。 |

---

## 2. 如何运行现有代码？(亲测可跑)

这里的脚本均已验证可打通“端到端验证环境”，**请确保使用依赖和虚拟环境正确（如`conda activate py310` ） 并在项目根目录下执行这些命令**。

### 2.1 运行 MNIST 半通用化实验（同口径对比）

通过 `run_leaf_expansion_mnist.py` 一个入口切换不同模：

1. **运行不含 LLM 的纯特征扩展**
   ```bash
   cd DeLTa-main
   python run_leaf_expansion_mnist.py --mode without_llm --save_summary
   ```

2. **运行假定 LLM (mock_llm) 的叶子扩展**
   ```bash
   cd DeLTa-main
   python run_leaf_expansion_mnist.py --mode llm_guided --relation_source mock_llm --save_summary
   ```

3. **运行真实在线 LLM (online_llm) 的尝试并自动落盘（需配好 local_openai_config.json）**
   ```bash
   cd DeLTa-main
   python tools/attempt_mnist_online_llm_same_metric.py
   ```

### 2.2 运行 Bank 非图像数据集迁移

1. **快速跑 bank 的 mock_llm 与 without_llm 基线**
   ```bash
   cd DeLTa-main
   python run_leaf_expansion_mnist.py --dataset bank --mode without_llm --save_summary
   python run_leaf_expansion_mnist.py --dataset bank --mode llm_guided --relation_source mock_llm --save_summary
   ```

2. **运行 bank 的 online_llm 同口径并生成逐叶子对比 (online vs mock)**
   ```bash
   cd DeLTa-main
   python tools/attempt_bank_online_llm_same_metric.py
   python tools/evaluate_bank_leaf_expansion_same_metric.py
   python tools/compare_bank_leaf_expansion_online_vs_mock.py
   ```

---

*（说明：这些运行方式确保我们现在能随时拉出同口径结果数据，不会出现实验一次性无法回查的问题。）*
