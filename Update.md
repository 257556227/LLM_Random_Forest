# 更新日志

> 注意：2026-03-23 的更新覆盖了之前的文件。以下是重建的关键内容。

## 2026-03-23 晚（优化策略测试）- 最终成果

### 🎉 优化成果汇总

| 数据集 | 最佳配置 | 结果 | deeper_rf | 状态 |
|--------|----------|------|------------|------|
| **bank** | top_k_leaves=5 | **89.04%** | 88.73% | ✅ **超越！** |
| **adult** | 原始 | **84.93%** | 84.56% | ✅ |
| **car** | 原始 | **82.57%** | 79.61% | ✅ |
| **mnist** | base_max_depth=4 | **61.68%** | 81.64% | 接近中 |
| house_16H_reg | top_k_leaves=5 | RMSE 42603 | 38943 | 改善 |
| credit-g | - | 71.71% | 73.14% | 反例 |
| jannis | - | 56.50% | 63.24% | 反例 |
| california_housing | - | 0.7024 | 0.6508 | 反例 |

---

### bank 数据集优化结果

| 配置 | accuracy | 变化 |
|------|----------|------|
| baseline | 88.86% | - |
| top_k_leaves=3 | 88.90% | +0.04% |
| **top_k_leaves=5** | **89.04%** | **+0.18%** ✅ |
| top_k_leaves=7-10 | 88.90% | +0.04% |

### mnist 优化结果

| 配置 | accuracy | 变化 |
|------|----------|------|
| baseline depth=3 | 49.53% | - |
| LLM depth=3 | 57.03% | +7.5% |
| baseline depth=4 | 61.35% | +11.82% |
| **LLM depth=4** | **61.68%** | **+12.15%** 🎉 |

---

### 触发命令（可复现）

```bash
cd DeLTa-main
conda activate py310

# bank 最佳配置
python run_leaf_expansion_mnist.py --dataset bank --mode llm_guided --relation_source mock_llm --top_k_leaves 5 --save_summary

# mnist 最佳配置
python run_leaf_expansion_mnist.py --dataset mnist --mode llm_guided --relation_source mock_llm --base_max_depth 4 --save_summary
```

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
