# Adult 实验台账（第一轮接入版）

| 日期 | 阶段 | 配置 | 结果 | 备注 |
|---|---|---|---|---|
| 2026-03-11 | 接入准备 | 审计工具 `audit_dataset_integration_readiness.py` | `ready_for_first_run = false` | 当前入口与模板存在，但数据层未落地 |
| 2026-03-11 | 数据准备 | `prepare_adult_dataset.py` | 成功生成 `example_datasets/adult/` | 产出 `N/C/y/info.json`，切分为 train/val/test |
| 2026-03-11 | 原型基线 | `run_leaf_expansion_mnist.py --dataset adult --mode without_llm` | `0.8467538848965052` | 相对 `prototype_baseline=0.8411645476322093` 有正向提升 |
| 2026-03-11 | 原型对照 | `relation_source=mock_llm` | `0.8440513481972852` | 第一轮 mock 未超过 without |
| 2026-03-11 | 原型对照 | `relation_source=online_llm` | `0.8492721577298692` | 当前最佳，已出现 `online_llm > without_llm > mock_llm` |
| 2026-03-11 | 审计刷新 | `audit_dataset_integration_readiness.py` | `ready_for_first_run = true` | `missing_items = []` |
| 2026-03-11 | 同口径摘要 | `evaluate_adult_leaf_expansion_same_metric.py` | 已生成统一摘要 | `online_llm_minus_mock_llm = 0.0052` |
| 2026-03-11 | 传统 RF 基线 | `run_randforest.py` (`DELTA_DATASETS=adult`) | `0.8585467723112831` | 产出 `adult_md20_ml50_tree15.npy` |
| 2026-03-11 | 传统 DeLTa 基线 | `train.py + ensemble.py` | `0.8602051471039862` | 本地规则 `model.llm_rule.adult.tree_full_md20_ml50_0` |
| 2026-03-11 | 正式协议摘要 | `evaluate_adult_formal_protocol.py` | 已生成正式协议摘要 | `online_llm` 超过 `Deeper RF=0.8456`，但低于传统主链 |

## 当前已确认入口
- 数据配置：`DeLTa-main/dataset_config.py`
- Prompt 模板：`DeLTa-main/llm/get_prompts/adult.py`
- 规则文件：`DeLTa-main/model/llm_rule/adult.py`
- 原型入口：`DeLTa-main/run_leaf_expansion_mnist.py`
- 就绪审计：`DeLTa-main/results/adult/adult_integration_readiness.json`
- 数据准备：`DeLTa-main/tools/prepare_adult_dataset.py`
- 同口径摘要：`DeLTa-main/tools/evaluate_adult_leaf_expansion_same_metric.py`

## 当前已落地产物
- `DeLTa-main/example_datasets/adult/`
- `DeLTa-main/results/adult/adult_integration_readiness.json`
- `DeLTa-main/results/adult/adult_leaf_expansion_same_metric_summary.json`
- `DeLTa-main/results/adult/adult_leaf_expansion_formal_protocol_summary.json`
- `DeLTa-main/results/adult/adult_md20_ml50_tree15.npy`
- `DeLTa-main/results/adult/e_RF_md20_ml50_tree15_adult_cart_0.log`
- `DeLTa-main/results/adult/leaf_expansion_without_llm_summary.json`
- `DeLTa-main/results/adult/llm_guided_leaf_expansion_summary.json`
- `DeLTa-main/results/adult/llm_guided_leaf_expansion_online_summary.json`

## 下一步
- 若继续深挖 `adult`，优先缩小 `online_llm` 与传统 `RF / DeLTa` 的约 `0.009` 到 `0.011` 差距
- 若转向扩数据集，优先把同一正式协议复制到 `credit-g`
- 保留当前结论：`adult` 已证明 `online_llm` 超过 `mock_llm`、`without_llm` 与 `Deeper RF`，但还没超过完整传统主链
