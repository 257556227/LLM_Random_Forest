# house_16H_reg 实验台账

## 1. 数据集定位
- 数据集：`house_16H_reg`
- 任务类型：回归
- 特征：16 个连续特征
- 目标：预测区域房价中位数
- 角色：第一个回归样板

## 2. 当前进展
- [x] 已确认 `example_datasets/house_16H_reg/` 与 `info.json` 存在
- [x] 已完成回归版 `leaf expansion` 原型兼容改造
- [x] 已跑通 `without_llm / mock_llm / online_llm` 三组原型
- [x] 已生成回归版同口径摘要
- [x] 已生成正式协议（`Deeper RF / without_llm / online_llm / traditional RF / traditional DeLTa`）摘要

## 3. 第一轮原型结果
- `prototype_baseline`: `RMSE = 43327.9766`, `R2 = 0.3317`
- `without_llm`: `RMSE = 42303.1098`, `R2 = 0.3630`
- `mock_llm`: `RMSE = 42843.3905`, `R2 = 0.3466`
- `online_llm`: `RMSE = 42044.3802`, `R2 = 0.3707`

## 4. 正式协议结果
- `deeper_rf`: `RMSE = 38943.8822`, `R2 = 0.4601`
- `without_llm`: `RMSE = 42303.1098`, `R2 = 0.3630`
- `online_llm`: `RMSE = 42044.3802`, `R2 = 0.3707`
- `traditional_rf`: `RMSE = 35230.4577`, `R2 = 0.5582`
- `traditional_delta(cart回退)`: `RMSE = 41434.7702`, `R2 = 0.3888`

## 5. 当前结论
- 回归版 `leaf expansion` 原型已经不是纸面兼容，而是能在真实数据上稳定降低 RMSE；
- `online_llm` 当前优于 `mock_llm` 与 `without_llm`，说明回归场景下真实在线排序也能从局部统计摘要中得到正向收益；
- 但正式协议显示它仍低于 `Deeper RF / traditional RF / traditional DeLTa(cart回退)`，因此当前应把 `house_16H_reg` 记为“回归原型有效但正式主对比仍未翻盘”的正式反例；
- 这仍然是重要里程碑，因为它标志着当前方法第一次从分类样板真正扩展到了回归样板。

## 6. 下一步
- 推进 `california_housing`，形成第二个回归样板；
- 复核“无 CUDA 环境下 `tabpfn -> cart` 自动回退”是否应推广为回归主链默认稳态；
- 对两份回归样板统一判断：这是不是当前方法在回归侧的系统性反例，还是只在 `house_16H_reg` 上受限。
