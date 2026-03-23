# house_16H_reg 模块映射（正式协议版）

## 1. 数据层
- 数据目录：`example_datasets/house_16H_reg/`
- 核心文件：`N_train.npy`、`N_val.npy`、`N_test.npy`、`y_train.npy`、`y_val.npy`、`y_test.npy`、`info.json`
- 当前特点：只有连续特征，没有类别特征；`info.json` 使用 `num_feature_intro / cat_feature_intro` 旧格式，现已由通用入口兼容读取

## 2. 原型入口
- 入口脚本：`run_leaf_expansion_mnist.py`
- 当前新增能力：
  - 自动根据 `task_type=regression` 切到 `DecisionTreeRegressor`
  - 原型指标从 `accuracy` 切到 `RMSE + R2`
  - 回归 leaf prompt 改为“连续目标区间分离 / 局部误差降低”表述

## 3. 叶子扩展核心模块
- `model/leaf_expansion/mnist_leaf_selector.py`
  - 新增回归叶子上下文：`predicted_value`、`target_summary`
  - 新增回归特征摘要：`variance_reduction`、`one_feature_train_rmse`
- `model/leaf_expansion/mnist_relation_provider.py`
  - 新增回归版 prompt，引导 LLM 优先选择能降低局部 RMSE 的阈值特征
- `model/leaf_expansion/mnist_leaf_expander.py`
  - 新增回归版局部树支持，兼容 `DecisionTreeRegressor`

## 4. 当前产物
- `results/house_16H_reg/leaf_expansion_without_llm_summary.json`
- `results/house_16H_reg/llm_guided_leaf_expansion_summary.json`
- `results/house_16H_reg/llm_guided_leaf_expansion_online_summary.json`
- `results/house_16H_reg/house_16H_reg_leaf_expansion_same_metric_summary.json`
- `results/house_16H_reg/house_16H_reg_leaf_expansion_formal_protocol_summary.json`
- `results/house_16H_reg/RF_md10_ml200_tree5_full_cart_0.npy`

## 5. 传统主链补充
- `run.py` / `run_ensemble.py` 在当前无 CUDA 环境下，会把回归主链中的 `tabpfn` 自动回退到 `cart`；
- 因此这轮正式协议里的 `traditional_delta` 是 `cart` 回退口径，产物为 `RF_md10_ml200_tree5_full_cart_0.npy`；
- 这一步的目的不是追求最优，而是先把回归正式协议变成“可稳定复现”的工程路径。

## 6. 当前结论
- 第一轮回归原型已经跑通，且 `online_llm` 在 `house_16H_reg` 上达到 `RMSE = 42044.3802`，优于 `without_llm = 42303.1098` 与 `mock_llm = 42843.3905`；
- 正式协议已补齐，但当前 `online_llm` 仍低于 `Deeper RF = 38943.8822`、`traditional_rf = 35230.4577` 与 `traditional_delta(cart回退) = 41434.7702`；
- 因此 `house_16H_reg` 当前应被记录为“回归正式反例”，但它仍然证明了 `LLM-Guided Leaf Expansion Forest` 已经具备从分类扩展到回归的真实样板能力。
