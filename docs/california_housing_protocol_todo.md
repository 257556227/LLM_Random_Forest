# california_housing 正式协议推进 TODO

## 目标
- 复用 house_16H_reg 正式协议推进流程，完成 california_housing 的正式协议样板。
- 验证 tabpfn->cart 自动回退逻辑在该数据集下的稳态适用性。
- 完善测试、文档、Update.md、summarize 小节。

## 步骤
1. 梳理/补全 california_housing 数据集的主链路（run.py/run_ensemble.py）配置与调用。
2. 检查/补充 tabpfn->cart 自动回退逻辑在该数据集下的适配。
3. 运行主链路，收集回归结果，若有异常及时修正。
4. 设计并补充正式协议测试用例（test/）。
5. 生成 formal protocol summary 脚本（tools/）。
6. 更新文档（docs/california_housing_integration_precheck.md、Update.md、summarize 等）。
7. 全面测试，确保所有链路与文档均通过。

## 参考 house_16H_reg 相关文件
- tools/evaluate_house_16H_reg_formal_protocol.py
- test/test_house_16H_reg_integration.py
- docs/house_16H_reg_integration_precheck.md
- DeLTa-main/results/house_16H_reg/
- DeLTa-main/reg_info/house_16H_reg.json

## 下一步
- 按上述 checklist 逐步推进，遇到分歧及时记录。
