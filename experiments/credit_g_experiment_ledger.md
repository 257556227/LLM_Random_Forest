# Credit-g 实验台账（正式协议版）

| 日期 | 阶段 | 配置 | 结果 | 备注 |
|---|---|---|---|---|
| 2026-03-11 | 接入准备 | 预检文档 `docs/credit_g_integration_precheck.md` | 已完成预检 | 当前入口与模板存在，但数据层和实验产物未落地 |
| 2026-03-11 | 数据准备 | `prepare_credit_g_dataset.py` | 成功生成 `example_datasets/credit-g/` | 产出 `N/C/y/info.json`，切分为 500/150/350 |
| 2026-03-11 | 原型基线 | `run_leaf_expansion_mnist.py --dataset credit-g --mode without_llm` | `0.7171428571428572` | 低于 `prototype_baseline=0.7285714285714285` |
| 2026-03-11 | 原型对照 | `relation_source=mock_llm` | `0.7285714285714285` | 把 `without_llm` 的下滑拉回到基线 |
| 2026-03-11 | 原型对照 | `relation_source=online_llm` | `0.7171428571428572` | 当前与 `without_llm` 持平，低于 `mock_llm` |
| 2026-03-11 | 同口径摘要 | `evaluate_credit_g_leaf_expansion_same_metric.py` | 已生成统一摘要 | `online_llm_minus_mock_llm = -0.0114` |
| 2026-03-11 | 传统 RF 基线 | `DELTA_DATASETS=credit-g python run_randforest.py` | `0.7428571428571429` | 产出 `credit-g_md15_ml20_tree20.npy` |
| 2026-03-11 | 传统 DeLTa 训练 | `DELTA_DATASETS=credit-g python run.py --num_answers 10` | 已完成 | 产出 `RF_md15_ml20_tree20_full_cart_*.npy` |
| 2026-03-11 | 传统 DeLTa 融合 | `DELTA_DATASETS=credit-g python run_ensemble.py --n_ensemble 0` | `0.7342857142857143` | `eta = 0.1` |
| 2026-03-11 | 正式协议摘要 | `evaluate_credit_g_formal_protocol.py` | 已生成正式协议 | `deeper_rf = 0.7314`，`online_llm = 0.7171` |
| 2026-03-23 | 网格调参 | depth=3, leaves=10, features=3 | `0.6971` | worse than baseline |
| 2026-03-23 | 网格调参 | depth=4, leaves=10, features=5 | `0.6971` | worse than baseline |
| 2026-03-23 | 网格调参 | depth=3, leaves=15, features=7 | `0.7171` | same as baseline |
| 2026-03-23 | 网格调参 | depth=5, leaves=20, features=5 | `0.7000` | worse than baseline |
| 2026-03-23 | 网格调参 | depth=4, leaves=5, features=3 | `0.6971` | worse than baseline |

## 当前已确认入口
- 数据配置：`DeLTa-main/dataset_config.py`
- Prompt 模板：`DeLTa-main/llm/get_prompts/credit-g.py`
- 规则文件：`DeLTa-main/model/llm_rule/credit-g.py`
- 原型入口：`DeLTa-main/run_leaf_expansion_mnist.py`
- 就绪审计：`DeLTa-main/tools/audit_dataset_integration_readiness.py`
- 数据准备：`DeLTa-main/tools/prepare_credit_g_dataset.py`
- 同口径摘要：`DeLTa-main/tools/evaluate_credit_g_leaf_expansion_same_metric.py`

## 当前已落地产物
- `DeLTa-main/example_datasets/credit-g/`
- `DeLTa-main/results/credit-g/credit-g_integration_readiness.json`
- `DeLTa-main/results/credit-g/credit_g_leaf_expansion_same_metric_summary.json`
- `DeLTa-main/results/credit-g/credit_g_leaf_expansion_formal_protocol_summary.json`
- `DeLTa-main/results/credit-g/leaf_expansion_without_llm_summary.json`
- `DeLTa-main/results/credit-g/llm_guided_leaf_expansion_summary.json`
- `DeLTa-main/results/credit-g/llm_guided_leaf_expansion_online_summary.json`
- `DeLTa-main/results/credit-g/credit-g_md15_ml20_tree20.npy`
- `DeLTa-main/results/credit-g/e_RF_md15_ml20_tree20_credit-g_cart_0.log`

## 下一步
- 重点排查为什么 `online_llm` 当前没有超过 `mock_llm`，且已经低于 `Deeper RF / RF / DeLTa`
- 优先尝试更贴近叶子局部统计的金融 prompt，而不是继续增加泛化业务故事
- 可以把 `credit-g` 与 `adult` 并列成“强语义表格正反样板”，再决定先回修哪个数据集
