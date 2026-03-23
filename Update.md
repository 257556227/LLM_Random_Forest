# 更新日志

> 注意：2026-03-23 的更新覆盖了之前的文件。以下是重建的关键内容。

## 2026-03-23 晚（优化策略测试）

### bank 数据集优化结果

| 配置 | accuracy | 变化 |
|------|----------|------|
| baseline | 88.86% | - |
| top_k_leaves=3 | 88.90% | +0.04% |
| top_k_leaves=5 | **89.04%** | **+0.18%** ✅ |
| top_k_leaves=10 | 88.90% | +0.04% |

**结论**：5 个叶子是最优配置

### credit-g 优化测试（均无效）

| 策略 | 结果 |
|------|------|
| 原始 LLM | 71.71% |
| Few-shot | 71.71% |
| Temperature=0 | 71.71% |
| 强制排序 | 71.71% |

---

## 2026-03-23（california_housing 数据补全 + 项目状态梳理）

### 本次目标
- 补全 california_housing 数据集（之前为空文件）
- 梳理所有8个数据集的实验状态
- 确认下一步推进方向

### 变更明细
1. 补全 california_housing 数据集：
   - 文件：DeLTa-main/example_datasets/california_housing/
   - 变更：通过 sklearn.datasets.fetch_california_housing 下载数据并保存为 .npy 文件
   - 结果：train(16560,8), val(2069,8), test(2011,8)

2. 验证所有数据集状态：
   - mnist: 已完成，online_llm 0.5622 > mock 0.5603
   - bank: 已完成，online_llm 0.8890 = mock
   - house_16H_reg: 已完成，反例
   - adult/credit-g/car/jannis: 已完成
   - california_housing: 数据就绪，待跑正式协议

### 实验结果汇总
| 数据集 | 类型 | online_llm | mock_llm | 状态 |
|--------|------|------------|----------|------|
| MNIST | 图像 | 0.5622 | 0.5603 | 成功 |
| Bank | 强语义 | 0.8890 | 0.8890 | 追平 |
| house_16H_reg | 回归 | RMSE=42044 | - | 反例 |

### 下一步
- california_housing 正式协议待跑

---

## 历史更新（需要从其他文档恢复）

更多历史更新记录请参考：
- docs/lookme.md - 执行清单
- docs/readme.md - 项目主文档
- experiments/ - 各数据集实验台账
