1. 因为Mnist数据集image就是不适合RF来做，10颗树根本包含不了784个特征（对吧？现在是包含了多少特征），所以拓展到尽量tabular data？记得抹去表头变量名和文件名，防止LLM学过。
2. 现在喂给LLM的内容是selector里面所有树的所有信息以及topK排序（这个其实也有一个问题，难道最后有的树还是3层，有的树4层吗？我以为都要统一）的叶子节点吗？这个prompt在哪里我没看到。如果没有喂tree原本的accuracy其实是不行的，threshold，就是LLM要先知道原本的accuracy再进一步补充下一个节点。包括这些参数都是在哪里调的，可以跟我讲讲不～
3. 你好像提到了prompt长度太长喂不进去的问题，可以1只喂features，那些判断可以不喂，2要比如10颗树RF那么每一个颗树上的features形成一个闭环，也就是所有10个特征都包含在10颗树里，且从任意一颗树出发可以连接到其它所有feature最终形成一个闭环？
4. withoutLLM  mockLLM 的区别是啥？不都是随机加一个节点变为4层树？
5. 调试结果的时候为啥有的时候好于mockLLM有点时候不好，具体调试的技术细节是？

1. MNIST 作为 RF 的数据集适用性、特征覆盖与 tabular 拓展
MNIST 不是 RF 的理想数据集，因为 784 维像素特征远大于树的分裂能力，10 棵树很难覆盖全部特征空间。实际每棵树只用到部分特征（可通过 RF 的 feature_importances_ 或 tree_.feature 统计），但远远达不到全覆盖。
目前已拓展到 tabular data，如 bank/adult/car/credit-g/jannis/house_16H_reg/california_housing 等，避免了 image 数据集的结构性劣势。
抹去表头变量名和文件名：prompt 构造时已做脱敏，防止 LLM 利用训练记忆（详见 mnist_relation_provider.py 的 prompt 构造逻辑）。
2. LLM 输入内容、叶子扩展层数、prompt 位置与参数调节
LLM 输入内容：每轮扩展时，selector 会收集所有树的所有叶子节点信息（如样本数、impurity、类别分布等），并对每个叶子选 topK 个候选特征，生成摘要（见 mnist_leaf_selector.py）。
层数问题：确实会出现有的树3层，有的树4层（甚至部分叶子没扩展），因为扩展是“局部生长”，不是全树统一加深。这样做是为了模拟“只在高杂质叶子下用 LLM 局部补结构”，而不是全树暴力加深。
prompt 位置：prompt 构造在 mnist_relation_provider.py 的 build_llm_leaf_prompt()，你可以直接查阅。
原本 accuracy/threshold：prompt 里会包含叶子的原始准确率、impurity、类别分布等（见 build_leaf_contexts() 和 prompt 构造），LLM 能看到这些局部性能指标。
参数调节：如 topK、min_leaf_samples、max_expansion_depth 等都在 run_leaf_expansion_mnist.py 的 argparse 入口和 config 文件里可调。
3. prompt 长度、只喂 features、闭环特征覆盖
prompt 长度问题：确实遇到过超长 prompt，解决方法有两种：
只喂 features：可以只给 LLM 叶子下的候选特征及其统计摘要，不给全部判断细节，减少 token 压力。
特征闭环覆盖：你说的“10棵树的 features 形成闭环”是指每棵树分裂的特征集合联合起来能覆盖全部特征，这个在 RF 里理论上成立，但实际每棵树只用到部分特征，且叶子扩展是局部的，不保证全闭环。
目前实现：每轮只对 topK 叶子和 topK 特征做扩展，prompt 只包含这些内容，避免超长。
4. withoutLLM 与 mockLLM 的区别
without_llm：只用启发式（如 impurity/samples）选叶子和特征，扩展时不引入任何 LLM 排序，完全靠统计指标。
mock_llm：用一个“假 LLM”排序器（见 mnist_relation_provider.py），本质是对候选特征做简单轮转或固定排序，模拟“有排序但无语义”的 baseline。
区别：without_llm 是纯启发式，mock_llm 是“有排序但无智能”，online_llm 才是真正用大模型排序。
5. onlineLLM 有时优于 mockLLM，有时不如，调试技术细节
原因：
prompt 设计不合理（如没给 enough 局部统计，LLM 无法判断）
LLM 理解 tabular 关系能力有限，或 prompt 太长被截断
叶子选择策略不佳，扩展到低价值叶子
数据集本身结构决定 LLM 排序难以超越启发式
调试细节：
对比每个叶子的扩展前后准确率/impurity，定位 LLM 排序是否真的带来增益
检查 prompt token 长度、内容是否包含关键统计指标
用 compare_*_leaf_expansion_online_vs_mock.py 工具分析每个叶子的增益分布
逐步引入 richer prompt（如 one_feature_train_accuracy、counterfactual check）提升 LLM 判断力
sweep topK、min_samples、prompt 内容等参数，找最优配置



参考文档位置（如 readme/lookme 有覆盖）：

readme.md：方法边界、实验协议、mock_llm 定义、已知问题
lookme.md：执行清单、失败分析、prompt 优化经验
mnist_relation_provider.py、mnist_leaf_selector.py：prompt 构造与参数调节
code_modification_mapping.md：核心代码结构与参数映射