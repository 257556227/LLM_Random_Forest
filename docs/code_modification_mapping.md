# 代码修改映射表 (Code Modification Mapping)

这份文档旨在记录 **LLM-Guided Leaf Expansion Forest** 原型开发过程中，哪些核心代码被新增或修改了，以及如何运行现有代码。

**最后更新日期: 2026年3月11日**

---

## 1. 核心目录与文件映射

| 类别 | 文件路径 | 核心功能与变动说明 |
|:---|:---|:---|
| **实验主入口** | `DeLTa-main/run_leaf_expansion_mnist.py` | **新增并持续增强**：统一实验入口，一键支持 `baseline_rf`, `deeper_rf`, `without_llm`, `llm_guided` 四种模式及不同数据集的切换；当前已新增 `local_max_depth`、`local_max_leaf_nodes`、`local_feature_subset_size`，支持从单特征局部树升级到多特征局部结构搜索。|
| **特征关系提取** (Relation) | `DeLTa-main/model/leaf_expansion/mnist_relation_provider.py` | **新增**：把关系提取独立模块化（对应 TODO 6）。负责向 LLM 构造包含 `task_intro`、`counterfactual check`、`one_feature_train_accuracy` 等特征描述的 Prompt，并严校验返回结果。 |
| **叶子上下文筛选** | `DeLTa-main/model/leaf_expansion/mnist_leaf_selector.py` | **新增**：负责在 RF 树的末端筛选出“高杂质”叶子，为候选特征计算信息增益、计算单特征准确率，并将这些前置信息准备给关系提取模块。 |
| **末端扩展逻辑** (Expansion) | `DeLTa-main/model/leaf_expansion/mnist_leaf_expander.py` | **新增并持续增强**：负责在指定的叶子节点下方挂载局部树；当前除 `top-k` 择优外，还支持在前 K 个候选特征中搜索多特征子集，执行双特征/多特征局部结构扩展。 |
| **统一结果与评估** | `DeLTa-main/tools/evaluate_mnist_same_metric_protocol.py`<br>`DeLTa-main/tools/evaluate_bank_leaf_expansion_same_metric.py`<br>`DeLTa-main/tools/evaluate_house_16H_reg_leaf_expansion_same_metric.py`<br>`DeLTa-main/tools/evaluate_house_16H_reg_formal_protocol.py` | **新增**：统一分类 / 回归同口径结果落盘逻辑，保障 Accuracy 或 RMSE / R2 对比不会漂移（对应 TODO 4）；当前已支持 `house_16H_reg` 的正式回归协议摘要。 |
| **局部分析工具** | `DeLTa-main/tools/compare_bank_leaf_expansion_online_vs_mock.py` | **新增**：针对 bank 数据集，逐个叶子比对 `online_llm` 和 `mock_llm` 以及局部训练/测试效果，用数据解释差异。 |
| **多数据集数据准备** | `DeLTa-main/tools/prepare_adult_dataset.py`<br>`DeLTa-main/tools/prepare_credit_g_dataset.py`<br>`DeLTa-main/tools/prepare_car_dataset.py`<br>`DeLTa-main/tools/prepare_jannis_dataset.py` | **新增**：将 `adult / credit-g / car / jannis` 统一转换到 DeLTa 所需的 `example_datasets/<dataset>/` 格式，并把 `task_intro / feature_intro / target_intro` 固化进 `info.json`。 |
| **正式协议汇总** | `DeLTa-main/tools/evaluate_adult_formal_protocol.py`<br>`DeLTa-main/tools/evaluate_credit_g_formal_protocol.py`<br>`DeLTa-main/tools/evaluate_car_formal_protocol.py`<br>`DeLTa-main/tools/evaluate_jannis_formal_protocol.py` | **新增**：统一输出 `Deeper RF / without_llm / online_llm / traditional RF / traditional DeLTa` 正式主对比协议摘要。 |
| **回归样板映射** | `DeLTa-main/docs/module_map_house_16H_reg.md`<br>`experiments/house_16H_reg_experiment_ledger.md` | **新增**：记录首个回归样板 `house_16H_reg` 的数据布局、回归版 leaf expansion 改造点、三组原型结果与正式协议结论。 |

---

## 2. 如何运行现有代码？(亲测可跑)

这里的脚本均已验证可打通“端到端验证环境”，**请确保使用 `conda activate py310` 并在项目根目录下执行这些命令**。

### 2.1 运行 MNIST 半通用化实验（同口径对比）

通过 `run_leaf_expansion_mnist.py` 一个入口切换不同模式（已实现 TODO 2）：

1. **运行不含 LLM 的纯特征扩展**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --mode without_llm --save_summary
   ```

2. **运行假定 LLM (mock_llm) 的叶子扩展**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --mode llm_guided --relation_source mock_llm --save_summary
   ```

3. **运行真实在线 LLM (online_llm) 的尝试并自动落盘（需配好 local_openai_config.json）**
   ```bash
   cd DeLTa-main
   conda run -n py310 python tools/attempt_mnist_online_llm_same_metric.py
   ```

4. **运行更强的局部结构搜索（适用于高维弱语义数据）**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset jannis --mode llm_guided --relation_source online_llm --top_k_leaves 4 --top_k_features_to_try 4 --local_feature_subset_size 2 --local_max_depth 2 --local_max_leaf_nodes 6 --save_summary --guided_output results/jannis/llm_guided_leaf_expansion_online_subset2_depth2_top4_summary.json
   ```

### 2.2 运行 Bank 非图像数据集迁移

1. **快速跑 bank 的 mock_llm 与 without_llm 基线**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset bank --mode without_llm --save_summary
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset bank --mode llm_guided --relation_source mock_llm --save_summary
   ```

2. **运行 bank 的 online_llm 同口径并生成逐叶子对比 (online vs mock)**
   ```bash
   cd DeLTa-main
   conda run -n py310 python tools/attempt_bank_online_llm_same_metric.py
   conda run -n py310 python tools/evaluate_bank_leaf_expansion_same_metric.py
   conda run -n py310 python tools/compare_bank_leaf_expansion_online_vs_mock.py
   ```

---

*（说明：这些运行方式确保我们现在能随时拉出同口径结果数据，不会出现实验一次性无法回查的问题。）*

---

## 3. 新增：第三数据集 `car` 的完整闭环入口

`car` 是当前第一个“低维纯类别多分类”正式样板，当前已跑到正式协议阶段。

1. **准备数据**
   ```bash
   cd DeLTa-main
   conda run -n py310 python tools/prepare_car_dataset.py
   conda run -n py310 python tools/audit_dataset_integration_readiness.py --dataset car
   ```

2. **跑三组原型**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset car --mode without_llm --save_summary --output results/car/leaf_expansion_without_llm_summary.json
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset car --mode llm_guided --relation_source mock_llm --save_summary --guided_output results/car/llm_guided_leaf_expansion_summary.json
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset car --mode llm_guided --relation_source online_llm --save_summary --guided_output results/car/llm_guided_leaf_expansion_online_summary.json
   conda run -n py310 python tools/evaluate_car_leaf_expansion_same_metric.py
   ```

3. **跑正式协议**
   ```bash
   cd DeLTa-main
   $env:DELTA_DATASETS='car'
   conda run -n py310 python run_randforest.py
   conda run -n py310 python run.py --num_answers 10
   conda run -n py310 python run_ensemble.py --n_ensemble 0
   conda run -n py310 python tools/evaluate_car_formal_protocol.py
   ```

4. **当前结论**
   - `online_llm = 0.8257`
   - `deeper_rf = 0.7961`
   - `traditional_rf = 0.8174`
   - `traditional_delta = 0.8470`

   这说明：`car` 当前已经是一个正向样板，且默认 `impurity_mass` 配置已经足够好，不需要继续在低收益超参上反复打磨。

---

## 4. 新增：第四数据集 `jannis` 的完整闭环入口

`jannis` 是当前第一个“高维弱语义多分类”正式样板，当前已跑到正式协议阶段。

1. **准备数据**
   ```bash
   cd DeLTa-main
   conda run -n py310 python tools/prepare_jannis_dataset.py
   conda run -n py310 python tools/audit_dataset_integration_readiness.py --dataset jannis
   ```

2. **跑三组原型**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset jannis --mode without_llm --save_summary --output results/jannis/leaf_expansion_without_llm_summary.json
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset jannis --mode llm_guided --relation_source mock_llm --save_summary --guided_output results/jannis/llm_guided_leaf_expansion_summary.json
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset jannis --mode llm_guided --relation_source online_llm --save_summary --guided_output results/jannis/llm_guided_leaf_expansion_online_summary.json
   conda run -n py310 python tools/evaluate_jannis_leaf_expansion_same_metric.py
   ```

3. **跑正式协议**
   ```bash
   cd DeLTa-main
   $env:DELTA_DATASETS='jannis'
   conda run -n py310 python run_randforest.py
   conda run -n py310 python run.py --num_answers 10
   conda run -n py310 python run_ensemble.py --n_ensemble 0
   conda run -n py310 python tools/evaluate_jannis_formal_protocol.py
   ```

4. **当前结论**
   - 默认 `online_llm = 0.5814`
   - 当前最好双特征局部树 `online_llm = 0.5857`
   - `deeper_rf = 0.6324`
   - `traditional_rf = 0.6488`
   - `traditional_delta = 0.6575`

   这说明：`jannis` 当前已经是一个正式样板，且“多特征局部结构搜索”比“更深单特征树”更有价值；但即便当前最好结果也仍低于 `deeper_rf`，后续若继续冲击主对照，更应该优先升级结构搜索与候选叶子策略，而不是继续细磨低收益超参。

---

## 5. 新增：第五数据集 `house_16H_reg` 的首个回归样板入口

`house_16H_reg` 是当前第一个“纯数值回归”样板，当前已完成三组原型、同口径摘要、正式协议、模块映射、台账与集成测试。

1. **直接跑三组回归原型**
   ```bash
   cd DeLTa-main
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset house_16H_reg --mode without_llm --save_summary --output results/house_16H_reg/leaf_expansion_without_llm_summary.json
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset house_16H_reg --mode llm_guided --relation_source mock_llm --save_summary --guided_output results/house_16H_reg/llm_guided_leaf_expansion_summary.json
   conda run -n py310 python run_leaf_expansion_mnist.py --dataset house_16H_reg --mode llm_guided --relation_source online_llm --save_summary --guided_output results/house_16H_reg/llm_guided_leaf_expansion_online_summary.json
   conda run -n py310 python tools/evaluate_house_16H_reg_leaf_expansion_same_metric.py
   ```

2. **跑正式协议**
   ```bash
   cd DeLTa-main
   $env:DELTA_DATASETS='house_16H_reg'
   conda run -n py310 python run_randforest.py
   conda run -n py310 python run.py --num_answers 1
   conda run -n py310 python run_ensemble.py --n_ensemble 0
   conda run -n py310 python tools/evaluate_house_16H_reg_formal_protocol.py
   ```

   说明：当前环境无 CUDA 时，`run.py / run_ensemble.py` 会自动把回归主链中的 `tabpfn` 回退成 `cart`，以避免 CPU `tabpfn` 的内存崩溃。

3. **当前第一轮回归结果**
   - `baseline_rmse = 43327.9766`
   - `without_llm_rmse = 42303.1098`
   - `mock_llm_rmse = 42843.3905`
   - `online_llm_rmse = 42044.3802`
   - `baseline_r2 = 0.3317`
   - `online_llm_r2 = 0.3707`

4. **当前正式协议结论**
   - `online_llm = 42044.3802`
   - `deeper_rf = 38943.8822`
   - `traditional_rf = 35230.4577`
   - `traditional_delta(cart回退) = 41434.7702`

   这说明：回归版 leaf expansion 已经跑通，且 `online_llm` 对原型链路确有正向收益；但在正式主对比下，`house_16H_reg` 当前仍是回归正式反例。下一步应把同一套流程迁到 `california_housing`，判断这是不是系统性现象。
