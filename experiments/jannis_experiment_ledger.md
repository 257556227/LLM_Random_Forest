# Jannis 实验台账（正式协议版）

| 日期 | 阶段 | 配置 | 结果 | 备注 |
|---|---|---|---|---|
| 2026-03-11 | 接入准备 | 预检文档 `docs/jannis_integration_precheck.md` | 已完成预检 | 当前入口存在，待补数据层、结果产物与正式协议 |
| 2026-03-11 | 数据准备 | `prepare_jannis_dataset.py` | 成功生成 `example_datasets/jannis/` | 产出 `N/y/info.json`，切分为 `53588 / 13398 / 16747` |
| 2026-03-11 | 原型基线 | `run_leaf_expansion_mnist.py --dataset jannis --mode without_llm` | `0.5639816086463247` | 低于 `prototype_baseline = 0.5666686570729086` |
| 2026-03-11 | 原型对照 | `relation_source=mock_llm` | `0.5651161402042156` | 略高于 `without_llm`，但仍低于原型基线 |
| 2026-03-11 | 原型对照 | `relation_source=online_llm` | `0.5814175673254911` | 明显高于 `baseline / without_llm / mock_llm` |
| 2026-03-11 | 同口径摘要 | `evaluate_jannis_leaf_expansion_same_metric.py` | 已生成统一摘要 | `online_llm_minus_mock_llm = +0.0163` |
| 2026-03-11 | 传统 RF 基线 | `DELTA_DATASETS=jannis python run_randforest.py` | `0.64877291455186` | 产出 `jannis_md20_ml50_tree20.npy` |
| 2026-03-11 | 传统 DeLTa 融合 | `DELTA_DATASETS=jannis python run_ensemble.py --n_ensemble 0` | `0.6574908938914432` | 当前仍是 jannis 最强主链结果 |
| 2026-03-11 | 正式协议摘要 | `evaluate_jannis_formal_protocol.py` | 已生成正式协议 | `online_llm` 强于三组原型，但仍低于 `Deeper RF / traditional RF / traditional DeLTa` |
| 2026-03-11 | 调优小扫 | `impurity / sample_count / top4 / topk4` | 默认配置最优或并列最优 | `impurity` 更差，`sample_count / top4 / topk4` 都未超过默认 |

## 当前已确认入口
- 数据配置：`DeLTa-main/dataset_config.py`
- Prompt 模板：`DeLTa-main/llm/get_prompts/jannis.py`
- 规则文件：`DeLTa-main/model/llm_rule/jannis.py`
- 原型入口：`DeLTa-main/run_leaf_expansion_mnist.py`
- 就绪审计：`DeLTa-main/tools/audit_dataset_integration_readiness.py`
- 数据准备：`DeLTa-main/tools/prepare_jannis_dataset.py`
- 同口径摘要：`DeLTa-main/tools/evaluate_jannis_leaf_expansion_same_metric.py`
- 正式协议：`DeLTa-main/tools/evaluate_jannis_formal_protocol.py`

## 当前已落地产物
- `DeLTa-main/example_datasets/jannis/`
- `DeLTa-main/results/jannis/jannis_integration_readiness.json`
- `DeLTa-main/results/jannis/leaf_expansion_without_llm_summary.json`
- `DeLTa-main/results/jannis/llm_guided_leaf_expansion_summary.json`
- `DeLTa-main/results/jannis/llm_guided_leaf_expansion_online_summary.json`
- `DeLTa-main/results/jannis/jannis_leaf_expansion_same_metric_summary.json`
- `DeLTa-main/results/jannis/jannis_md20_ml50_tree20.npy`
- `DeLTa-main/results/jannis/e_RF_md20_ml50_tree20_jannis_cart_0.log`
- `DeLTa-main/results/jannis/jannis_leaf_expansion_formal_protocol_summary.json`

## 下一步
- 把 `jannis` 作为“高维弱语义多分类样板”写回通用化方法论
- 下一站优先切 `house_16H_reg`，开始验证回归路径是否也能按同一套路收口