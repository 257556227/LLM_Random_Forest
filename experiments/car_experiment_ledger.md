# Car 实验台账（正式协议版）

| 日期 | 阶段 | 配置 | 结果 | 备注 |
|---|---|---|---|---|
| 2026-03-11 | 接入准备 | 预检文档 `docs/car_integration_precheck.md` | 已完成预检 | 当前入口与模板存在，待补数据层与正式实验产物 |
| 2026-03-11 | 数据准备 | `prepare_car_dataset.py` | 成功生成 `example_datasets/car/` | 产出 `C/y/info.json`，切分为 `864 / 256 / 608` |
| 2026-03-11 | 原型基线 | `run_leaf_expansion_mnist.py --dataset car --mode without_llm` | `0.8042763157894737` | 高于 `prototype_baseline = 0.7631578947368421` |
| 2026-03-11 | 原型对照 | `relation_source=mock_llm` | `0.7911184210526315` | 低于 `without_llm` |
| 2026-03-11 | 原型对照 | `relation_source=online_llm` | `0.8256578947368421` | 高于 `without_llm / mock_llm` |
| 2026-03-11 | 同口径摘要 | `evaluate_car_leaf_expansion_same_metric.py` | 已生成统一摘要 | `online_llm_minus_mock_llm = +0.0345` |
| 2026-03-11 | 传统 RF 基线 | `DELTA_DATASETS=car python run_randforest.py` | `0.8174342105263158` | 产出 `car_md10_ml10_tree20.npy` |
| 2026-03-11 | 传统 DeLTa 融合 | `DELTA_DATASETS=car python run_ensemble.py --n_ensemble 0` | `0.8470394736842105` | 当前仍是 car 最强主链结果 |
| 2026-03-11 | 正式协议摘要 | `evaluate_car_formal_protocol.py` | 已生成正式协议 | `online_llm > Deeper RF / traditional RF`，但 `< traditional DeLTa` |
| 2026-03-11 | 调优小扫 | `impurity / sample_count / top4 / topk4` | 默认配置最优 | `impurity_mass` 与 `impurity` 等价，`sample_count` 更差 |

## 当前已确认入口
- 数据配置：`DeLTa-main/dataset_config.py`
- Prompt 模板：`DeLTa-main/llm/get_prompts/car.py`
- 规则文件：`DeLTa-main/model/llm_rule/car.py`
- 原型入口：`DeLTa-main/run_leaf_expansion_mnist.py`
- 就绪审计：`DeLTa-main/tools/audit_dataset_integration_readiness.py`
- 数据准备：`DeLTa-main/tools/prepare_car_dataset.py`
- 同口径摘要：`DeLTa-main/tools/evaluate_car_leaf_expansion_same_metric.py`
- 正式协议：`DeLTa-main/tools/evaluate_car_formal_protocol.py`

## 当前已落地产物
- `DeLTa-main/example_datasets/car/`
- `DeLTa-main/results/car/car_integration_readiness.json`
- `DeLTa-main/results/car/leaf_expansion_without_llm_summary.json`
- `DeLTa-main/results/car/llm_guided_leaf_expansion_summary.json`
- `DeLTa-main/results/car/llm_guided_leaf_expansion_online_summary.json`
- `DeLTa-main/results/car/car_leaf_expansion_same_metric_summary.json`
- `DeLTa-main/results/car/car_md10_ml10_tree20.npy`
- `DeLTa-main/results/car/e_RF_md10_ml10_tree20_car_cart_0.log`
- `DeLTa-main/results/car/car_leaf_expansion_formal_protocol_summary.json`

## 下一步
- 以 `car` 为“低维纯类别多分类样板”写回通用化方法论
- 下一站优先切 `jannis`，验证高维弱语义多分类是否也能稳定复用当前流程
