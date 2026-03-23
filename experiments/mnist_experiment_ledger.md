# MNIST 实验台账（DeLTa）

| 日期 | 配置 | 在线答案数 | n_ensemble | eta | Fused Acc | 备注 |
|---|---|---:|---:|---:|---:|---|
| 2026-02-18 | md8/ml20/tree3 | 1 | 0 | 0.2 | 0.8285 | 单答案口径，偏低 |
| 2026-02-18 | md8/ml20/tree3 | 3 | 2 | 0.2 | 0.8969 | 多答案后回升 |
| 2026-02-18 | md8/ml20/tree3 | 6 | 5 | 0.2 | 0.9158 | 继续回升 |
| 2026-02-19 | md8/ml20/tree3 | 10 | 9 | 0.2 | 0.9260 | 当前稳定高效果参数 |
| 2026-02-19 | Prompt A=mnist_stable.py | 10 | 9 | 0.2 | 0.9198 | A/B 对照基线 |
| 2026-02-19 | Prompt B=mnist_ab_b.py | 10 | 9 | 0.2 | 0.9154 | 低于A，未采用 |
| 2026-02-20 | Prompt C=mnist_prompt12_aligned.py | 10 | 9 | 0.2 | 0.9153 | 网络可达后完成同口径评估 |

## 稳定复现实验命令要点
- 在线查询：`run_get_answer.py --num_queries 10 --request_interval 2 --max_completion_tokens 2200 --normalize_tree_output 1`
- 训练：`run.py --num_answers 10`
- 融合：`run_ensemble.py --n_ensemble 9 --eta 0.2`

## 下一步
- 在固定上述参数下，开始 Prompt A/B（仅替换 `mnist_stable.py` 文案），每版至少 10 次统计均值。

## A/B 对照结论（2026-02-19）
- 对照日志：
	- A: `DeLTa-main/results/mnist/e_ab_A.log`
	- B: `DeLTa-main/results/mnist/e_ab_B.log`
- 结果：A(0.9198) > B(0.9154)，差值 +0.0044。
- 结论：当前保留 `mnist_stable.py` 作为线上稳定模板。

## Prompt1/Prompt2 对齐实验说明（2026-02-20）
- 新模板：`DeLTa-main/llm/get_prompts/mnist_prompt12_aligned.py`
- 设计点：加入“相似性/多样性平衡、跨特征关联、routing-style 分层”约束。
- 结果日志：`DeLTa-main/results/mnist/e_ab_C.log`
- 结果：C(0.9153)，低于 A(0.9198)，接近 B(0.9154)。
- 结论：在当前固定口径下，继续保留 A（`mnist_stable.py`）作为首选模板，C 作为备选探索方向。

## 四组协议库存摘要（2026-03-08）
- 汇总脚本：`DeLTa-main/tools/build_mnist_protocol_summary.py`
- 统一 runner：`DeLTa-main/tools/run_mnist_four_way_protocol.py --dry_run`
- 协议配置：`DeLTa-main/configs/leaf_expansion/mnist_four_way_protocol.json`
- 汇总结果：`DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`
- 当前状态：`strict_same_metric_protocol_ready = false`
- 已汇总四组：
	- `baseline_rf`：`e_RF_md8_ml20_tree3_mnist_cart_9.log`，`fused_accuracy_mean = 0.9153`
	- `deeper_rf`：`e_RF_md20_ml100_tree15_mnist_cart_9.log`，`fused_accuracy_mean = 0.9421`
	- `without_llm`：`leaf_expansion_without_llm_summary.json`，`0.4953 -> 0.5413`
	- `llm_guided`：`llm_guided_leaf_expansion_summary.json`，`0.4953 -> 0.5603`
- 当前说明：
	- `baseline_rf / deeper_rf` 来自 DeLTa 主链的 fused eval；
	- `without_llm / llm_guided` 来自叶子扩展原型的 single-model eval；
	- 因此这份摘要当前用于阶段一收口和产物盘点，不作为最终同口径论文结论。

## 原型协议安全复跑（2026-03-08）
- 执行入口：`conda run -n py310 python tools/run_mnist_four_way_protocol.py --stage without_llm --stage llm_guided --stage build_summary`
- 本轮实际刷新产物：
	- `DeLTa-main/results/mnist/leaf_expansion_without_llm_summary.json`
	- `DeLTa-main/results/mnist/llm_guided_leaf_expansion_summary.json`
	- `DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`
- 本轮结果：
	- `without_llm`：`0.4953 -> 0.5413`
	- `llm_guided(mock_llm)`：`0.4953 -> 0.5603`
- 结论：
	- 统一 runner 已不仅能 `dry-run`，也已完成一轮真实原型侧复跑；
	- 当前原型侧结果与之前库存摘要保持一致；
	- 还未进入严格同口径重跑的仍然是 `baseline_rf / deeper_rf` 两组主链实验。

## 主链协议复跑（2026-03-08）
- 执行入口：`conda run -n py310 python tools/run_mnist_four_way_protocol.py --stage baseline_rf --stage deeper_rf --stage build_summary`
- 本轮实际刷新产物：
	- `DeLTa-main/results/mnist/mnist_md8_ml20_tree3.npy`
	- `DeLTa-main/results/mnist/e_RF_md8_ml20_tree3_mnist_cart_9.log`
	- `DeLTa-main/results/mnist/mnist_md20_ml100_tree15.npy`
	- `DeLTa-main/results/mnist/e_RF_md20_ml100_tree15_mnist_cart_9.log`
	- `DeLTa-main/results/mnist/mnist_leaf_expansion_protocol_summary.json`
- 本轮结果：
	- `baseline_rf`：`RF Accuracy MEAN = 0.6908`，`Fused Accuracy MEAN = 0.9153`
	- `deeper_rf`：`RF Accuracy MEAN = 0.9025`，`Fused Accuracy MEAN = 0.9421`
- 结论：
	- 统一 runner 已经完成四组库存协议全部阶段的真实执行；
	- 当前“四组协议还未严格同口径”的问题，已经不再是“有没有统一执行入口”，而是“评估 scope 还没有收敛到同一协议”。

## 同口径摘要生成（2026-03-08）
- 执行入口：`conda run -n py310 python tools/evaluate_mnist_same_metric_protocol.py`
- 结果文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`
- 当前统一口径：`metric_scope = same_test_accuracy_eval`
- 当前结果：
	- `baseline_rf = 0.6908`
	- `deeper_rf = 0.9025`
	- `without_llm = 0.5413`
	- `llm_guided(mock_llm) = 0.5603`
- 结论：
	- 当前已经拥有一份真正统一 metric scope 的四组协议摘要；
	- 库存摘要继续负责历史产物盘点，同口径摘要负责 apples-to-apples 的 test accuracy 对照。

## online_llm 同口径正式尝试（2026-03-08）
- 执行入口：`conda run -n py310 python tools/attempt_mnist_online_llm_same_metric.py`
- 结果文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_online_llm_attempt.json`
- 同步摘要：`DeLTa-main/results/mnist/mnist_leaf_expansion_same_metric_summary.json`
- 当前状态：`status = success`
- 当前环境：
	- `OPENAI_API_KEY` 已存在；
	- `OPENAI_BASE_URL` 已存在；
	- `OPENAI_MODEL` 当前由本地配置文件提供。
- 当前结果：
	- `baseline_accuracy = 0.4953`
	- `test_accuracy = 0.5622`
	- `llm_model = gpt-4o`
	- 同口径摘要已新增 `experiments.online_llm`，并记录 `online_llm_minus_mock_llm = +0.0019`。
- 本轮实现增量：
	- prompt 新增候选特征的 `pixel`、`information_gain`、`class_mean_spread`、`predicted_class_mean_gap`；
	- online 响应若出现候选外 id 或重复 id，会直接重试；
	- 扩展器不再只吃 top-1，而是在 top-k 候选中试分裂后再选最优单特征。
- 历史排查结论：
	- 早先一轮确实卡在 `codemirror.codes` 的 Cloudflare 拦截；
	- 切到官方端点后也曾出现 `401 invalid_api_key`；
	- 本轮用户更新本地私有配置后，真实 `online_llm` 已成功跑通，因此上述问题已降级为历史排查记录。
- 结论：
	- 当前主链和原型代码不再处于“是否真的执行过在线尝试”的不确定状态；
	- 当前已从“为什么 online 低于 mock”切换到“当前 `+0.0019` 的微弱反超是否稳定”。

## oracle 单特征上界分析（2026-03-08）
- 执行入口：`conda run -n py310 python tools/analyze_mnist_leaf_oracle.py`
- 结果文件：`DeLTa-main/results/mnist/mnist_leaf_expansion_oracle_analysis.json`
- 当前结论：
	- `leaf 8` 的 oracle 最优单特征为 `347`，局部 `test_accuracy = 0.5196`；
	- `online_llm` 当前选中 `347`，`oracle_gap = 0.0`；
	- `mock_llm` 当前选中 `374`，`oracle_gap = 0.0131`；
	- `leaf 9` 的多个候选单特征局部效果并列，因此三条路线在该叶子上几乎打平。

## online_llm 五次重复稳态验证（2026-03-08）
- 执行入口：`conda run -n py310 python tools/repeat_mnist_online_llm_same_metric.py --repeats 5`
- 结果文件：`DeLTa-main/results/mnist/mnist_online_llm_repeatability_summary.json`
- 当前结果：
	- `repeats = 5`
	- `success_count = 5`
	- 5 次 `test_accuracy` 全部为 `0.5622`
	- `online_accuracy_std = 0.0`
	- `wins_over_mock_count = 5`
- 结论：
	- 当前 `online_llm = 0.5622` 相对 `mock_llm = 0.5603` 的 `+0.0019` 已经不是 3 次小样本波动，而是在当前 5 次重复下保持稳定。

## 下一步
- 先分析为什么当前这条已稳定的 `MNIST` 微弱增益没有直接迁到 `bank`；
- 再决定是否继续补 `bank` 的 task/target 语义增强，还是扩到更多数据集。
