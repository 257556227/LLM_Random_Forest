**Augmenting Decision Trees with Learned Feature Dependence from Large Language Models**

This project proceeds as follows: on each dataset (e.g., MNIST, USPS, Pendigits, Letter, Optical Recognition, Connect-4, Protein, and SenseIT), we first use standard CART / Random Forest procedures to generate a set of shallow decision trees from data with roughly 200 observations and 10 features (e.g., depth-3 trees using 3–4 features per tree, with about 10–20 trees for structure extraction). The splitting structures of these trees are then summarized and provided to a large language model to learn feature dependence patterns. During the construction of base learners, we first augment each tree by introducing one additional, LLM-suggested feature at the leaf nodes, and further explore extending local tree structures into graph-based representations (tree-to-graph) to enhance the expressive power of individual base learners. Finally, 100–200 enhanced base learners are aggregated in an ensemble, and their performance is evaluated against standard CART and Random Forest baselines on the above benchmark datasets.

 

我过了下这两篇论文的方法，其中第一个问题的json形式在DeLTa框架可以参考。他就是将决策树的规则转化为语义丰富的语法，这部分可以借鉴，不过它里面提到的节点内样本距离光看描述不太能确定，要做的话得看看它的源代码来作为参考（我看有开源）

同时第二个问题的话，DeLTa那篇也有启示，数据集数量可控的前提下，给每个数据集向 LLM 发送一次查询，api还是可以成本可控的。至于他说的本地部署32B模型，应该需要至少A800的GPU来推理会更好，如果只跑一个数据集的话，还不如api来的划算

最后第三个问题，也有Delta的残差修正可以缓解规避，这点效仿下也可以，整体来看它对于LLM不敏感，不过闭环需要根据数据集的标签类型来切换损失函数，我可以看看它目前支持多少，如果这个支持的足够多，或者好修改的话，新加数据集的难度会比较低，重点就是确保 Prompt Template 能够自动读取数据集的维度和初始树逻辑，做通用化会初始会比之前想的要多耗费不少时间，但后面加起来会比较方便

不过你提到的tree-to-graph应该是TnT 这篇论文提出来的把，这个就不只是简单的规则叠加，而是相当于在之前LLM总结的基础上做一个结构增强，它的算法逻辑是通用的，新数据集只要适配已有我们搭建的框架，这一点不会造成多少不同数据集之间的切换成本

 

 

**一、整体一句话概括**

**核心思想**：
 我们不改变 Random Forest / Tree Ensemble 的整体框架，而是**用大语言模型（****LLM****）增强每一棵** **base learner****（决策树）的构造方式**，让单棵树能捕捉到更多跨特征的依赖关系，从而在几乎不增加计算复杂度的前提下，提升整体预测性能。

这句话非常重要，**这是审稿人****/****合作者第一眼要懂的东西**。

 

**二、先澄清你刚才提到的几个****“****你有点不确定但方向是对的****”****的地方**

**1️****⃣** **你对传统** **Decision Tree / Random Forest** **的理解：大体是对的**

传统流程（以 Random Forest 为例）：

·       有 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAeCAMAAAArQN94AAAAAXNSR0IArs4c6QAAAEtQTFRFAAAAAAAAAAA6ADqQAGa2OgAAOgBmOmZmOpDbZgA6ZgBmZpCQZrbbZrb/kDoAkNvbkNv/tmYAtv//25A62////7Zm/9uQ//+2///bhys7OwAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAbUlEQVQoU81OSRKAMAgL1rpUrQtd/P9LhdbpF5QDTCAJAX5akcgEgIn6kpCpO2SculQ42kmhNuD2q5dDnpUCZHexsNNwFRhNSMKO1QjcC9+EJt1k1e1ehkrVItnFNam6lyjtO1P9+mZLtko/qge+/gSLrcZdAgAAAABJRU5ErkJggg==)个样本（比如 200 个）

·       有 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAeCAMAAAAiq38CAAAAAXNSR0IArs4c6QAAAFpQTFRFAAAAAAAAAAA6AABmADqQAGa2OgAAOgA6OmaQOma2OpDbZgAAZjoAZrbbZrb/kDoAkJBmkNv/tmYAtmY6tv//25A625Bm27Zm27aQ2////7Zm/9uQ//+2///bFCJCUwAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAZElEQVQoU82OSRKAIAwEAwguiAruLP//pgP4AW/2ITVJ9RQQ/Yu0Mkmn4hPRYb1YDDmx4Ys7GzC4RXTN/cY041iFoNCIWmL32SoDLUOXyhK5bmTc5BQ11Epos1WohYJjPV74ygP1sQPsxueyvgAAAABJRU5ErkJggg==)个特征（比如 10 个）

·         每一棵树：

o        **Bootstrap** 抽样（有放回抽样样本）

o        **随机选特征子集**（比如每个节点只看 3–4 个特征）

o        树通常很浅（3–5 层）

·         每一棵树 **只学到****“****局部、低维****”****的特征关系**

·         Forest 的能力来自于：

o        多棵树

o        decorrelation（不相关）

👉 你的直觉是对的：
 **单棵树很****“****弱****”****，信息量是受限的**。

 

**三、你的方法想解决什么****“****根本问题****”****？**

**传统** **Tree-based** **方法的一个结构性限制**

**限制**：
 决策树是“逐层贪心 + 局部视野”的模型
 → 很难显式捕捉 **多变量之间的复杂依赖结构**

比如：

·         ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAeCAMAAAD5ENUgAAAAAXNSR0IArs4c6QAAAGNQTFRFAAAAAAAAAAA6AABmADpmADqQAGa2OgAAOgA6OjqQOma2OpC2OpDbZgAAZgA6ZjoAZrb/kDoAkLbbkNv/tmYAtpBmtrZmtv//25A627Zm29u22////7Zm/9uQ/9u2//+2///b2KGS0gAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAkklEQVQoU91QzRqCMAxrnYg/TBiidDrc3v8pacelGzeP5tQvzZekBfgb+BbNnBxip06aDg+AgLeP4pYzc9OpODy5DuiiVbym5lVTsLRmrl9GWJrxPphnrePYaHO3d494lSHagZs0HBvvI3jpmpx4kYyC3NUhS749Hrdk0gduMp/tCtCeCkwFTlSIFhklt7P6gVgBD0cGSK2e2F4AAAAASUVORK5CYII=)和 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAeCAMAAAD5ENUgAAAAAXNSR0IArs4c6QAAAHVQTFRFAAAAAAAAAAA6AABmADo6ADpmADqQAGa2OgAAOgA6OjoAOjqQOpC2OpDbZgAAZgA6ZjoAZpDbZrbbZrb/kDoAkDo6kGY6kLbbkNv/tmYAtpBmtrZmtv//25A625Bm27Zm27aQ2////7Zm/9uQ/9u2//+2///bFR1tOgAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAmUlEQVQoU92SzRLCIAyEN0Vb/Gm1ilq0tNoW3v8RDYwHwKsnc8psMrtfGIC/qVGS6J0iqqOTdNEBE+1fkTZvWNNVcrhTNcwu3uKxKR+5hFmKPn8yQ6kZzydxz/c41jaBbdhS0frGNmcmKTnWXW4YPatT3sv41ldgVcQry5HWIXm5nnIATasvDXhKdo/KHjrGTzUM8sPy23/wBnEeB9lEMMlCAAAAAElFTkSuQmCC)单独看都没什么用

·       但 **一起出现时** 对 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAeCAMAAADNaRQ8AAAAAXNSR0IArs4c6QAAAEVQTFRFAAAAAAAAAABmADqQAGa2OgAAOjpmOpC2OpDbZgAAZjo6Zrb/kDoAkDo6kNv/tmYAtv//25A62////7Zm/9uQ//+2///b3PyX0gAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAWklEQVQoU8WO2xKAIAhEd8suZhdN6f8/NRJn+oM6TwywcICfEE8G4CAdkDmrhnh3AteqDaReSx13EWWM1TEzyLKbrvhp07Rx1HSrNdCwQ0YZ3pX0/DUSq8QH3LzbAm3XlpDuAAAAAElFTkSuQmCC)非常关键

·         普通 CART 很可能永远 split 不出来

 

**四、你们的方法：分成「两大阶段** **+** **两个可选分支」**

我先给你一个**总结构图（逻辑层面）**，再逐步展开。

 

**Stage 0****：问题设置**

·         数据：

o              ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAeCAMAAAArQN94AAAAAXNSR0IArs4c6QAAAEtQTFRFAAAAAAAAAAA6ADqQAGa2OgAAOgBmOmZmOpDbZgA6ZgBmZpCQZrbbZrb/kDoAkNvbkNv/tmYAtv//25A62////7Zm/9uQ//+2///bhys7OwAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAbUlEQVQoU81OSRKAMAgL1rpUrQtd/P9LhdbpF5QDTCAJAX5akcgEgIn6kpCpO2SculQ42kmhNuD2q5dDnpUCZHexsNNwFRhNSMKO1QjcC9+EJt1k1e1ehkrVItnFNam6lyjtO1P9+mZLtko/qge+/gSLrcZdAgAAAABJRU5ErkJggg==)个观测（如 200）

o              ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAeCAMAAAAiq38CAAAAAXNSR0IArs4c6QAAAFpQTFRFAAAAAAAAAAA6AABmADqQAGa2OgAAOgA6OmaQOma2OpDbZgAAZjoAZrbbZrb/kDoAkJBmkNv/tmYAtmY6tv//25A625Bm27Zm27aQ2////7Zm/9uQ//+2///bFCJCUwAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAZElEQVQoU82OSRKAIAwEAwguiAruLP//pgP4AW/2ITVJ9RQQ/Yu0Mkmn4hPRYb1YDDmx4Ys7GzC4RXTN/cY041iFoNCIWmL32SoDLUOXyhK5bmTc5BQ11Epos1WohYJjPV74ygP1sQPsxueyvgAAAABJRU5ErkJggg==)个特征（如 10）

·         目标：

o        分类或回归

·         Benchmark：

o        CART

o        Random Forest

o        （可选）XGBoost

 

**五、****Stage 1****：用** **LLM** **学「特征依赖结构」（这是第一个创新核心）**

**🎯** **目标**

不是让 LLM **直接预测** **Y**，
 而是让它学：

**“****这些特征之间是怎么一起工作的？****”**

 

**1️****⃣** **输入给** **LLM** **的不是原始大数据，而是****“****树级别的信息****”**

你这里的想法非常对，而且很聪明：

**我们不是把** **200×10** **的原始表格直接喂给** **GPT**
 而是先用传统方法“压缩”成结构性信息

 

**方法** **A****：基于****“****随机低维树****”****的结构学习（你提到的第一种）**

流程：

1. 用传统方式：

o        随机抽样

o        随机选特征

o        生成一批 **低维、浅层决策树**（比如 10 棵）

2. 每棵树只涉及：

o        3–4 个特征

3. 把这些树的：

o        split 顺序

o        使用的特征

o        叶子条件
 **转成结构化文本** **/ JSON**

4. 喂给 LLM，让它回答：

o        哪些特征经常一起出现？

o        哪些特征组合决定预测？

o        是否存在条件依赖关系？

👉 本质上：
 **LLM** **在****“****树的集合上做结构归纳****”**

 

**方法** **B****：全变量相关** **/** **条件依赖分析（你提到的第二种）**

这是一个更系统的版本：

·         对每个 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAeCAMAAAD95QUdAAAAAXNSR0IArs4c6QAAAGBQTFRFAAAAAAAAAAA6AABmADpmADqQAGa2OgAAOgA6OgBmOjqQOpC2OpDbZgAAZjoAZrbbZrb/kDoAkNv/tmYAtpBmtrZmtv//25A625CQ27Zm2////7Zm/9uQ/9u2//+2///bi8vuswAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAiUlEQVQoU9VRyxKCMAxMqIDKQyJKhBb6/3/JVkagnLmYQ6azm+5uWqK/rD5n03lhLtf4bfIkslwMK+KuQNpst6CXkvS+TYDS9B0D5HLTxY+ivBcBZ80rnoHVVC1ZNA36U9XA+3tcykvQ0BDSS+gkDHqs+QI3dztYkj1YQm3b9SfZfOJrUH6c958zIaEF9w/vO0oAAAAASUVORK5CYII=)：

o                   用其余 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAeCAMAAADqx5XUAAAAAXNSR0IArs4c6QAAAGBQTFRFAAAAAAAAAAA6AABmADpmADqQAGa2OgAAOgA6OgBmOjqQOpC2OpDbZgAAZjoAZrbbZrb/kDoAkNv/tmYAtpBmtrZmtv//25A625CQ27Zm2////7Zm/9uQ/9u2//+2///bi8vuswAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAlElEQVQ4T+1SyxLCIAxMxFofRRurRQXJ//+lgR6cMsnVk3sKk112EwD4w9jAs0c3MyEOCmHaXAEinl5KL+2lN+3Ua5kGCEdNJfTQ3a0WpN7N1lMF1M2EH93N0knM7NvZQleyZT/KBLVswVS8Qhn/C6Z6JhTJ+4zbVdJ0aIJnjxXiEc3g4q/tffFkGh/WNiTL5fd/+gM7kAY4Fh+mogAAAABJRU5ErkJggg==)作为输入

o        分析：

§                             哪些变量对 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAeCAMAAAD95QUdAAAAAXNSR0IArs4c6QAAAGBQTFRFAAAAAAAAAAA6AABmADpmADqQAGa2OgAAOgA6OgBmOjqQOpC2OpDbZgAAZjoAZrbbZrb/kDoAkNv/tmYAtpBmtrZmtv//25A625CQ27Zm2////7Zm/9uQ/9u2//+2///bi8vuswAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAiUlEQVQoU9VRyxKCMAxMqIDKQyJKhBb6/3/JVkagnLmYQ6azm+5uWqK/rD5n03lhLtf4bfIkslwMK+KuQNpst6CXkvS+TYDS9B0D5HLTxY+ivBcBZ80rnoHVVC1ZNA36U9XA+3tcykvQ0BDSS+gkDHqs+QI3dztYkj1YQm3b9SfZfOJrUH6c958zIaEF9w/vO0oAAAAASUVORK5CYII=)重要

§        条件关系如何变化

·         把这些关系统一交给 LLM

·         让 LLM 总结：

o        高阶交互

o        非线性依赖

o        潜在因果/逻辑结构（不强调“因果”，强调“dependence”）

 

**🎯** **Stage 1** **的输出（非常关键）**

**一个** **Feature Dependence Graph / Structure**

形式可以是：

·         Graph（节点=特征，边=依赖）

·         Rule set

·         Hierarchical relation

·         JSON 结构描述

这是后面一切的“知识库 / 记忆”。

 

**六、****Stage 2****：用** **LLM** **增强「单棵树的构造方式」（第二个创新核心）**

现在重点来了：
 **不是替换** **forest****，而是增强** **base learner**

 

**基础版本（你提到的****“****最简单但最稳妥****”****的）**

**🔹** **Leaf-level augmentation****（强烈推荐作为主方法）**

·         仍然用 CART 方式生长树

·         树深度仍然很小（比如 3 层）

·         **但在叶子节点处**：

o        不直接输出预测

o        而是：

§        让 LLM 根据 dependence graph

§        引入 **额外的特征** **/** **规则组合**

·         相当于：

o        每棵树“看起来”还是一棵树

o        但每个叶子节点是 **增强版的局部模型**

👉 这点你说得非常对：

增加一层 ≠ 指数级增加复杂度
 但可以显著提高单棵树的信息容量

 

**进阶版本（你提到的** **graph /** **非树结构）**

这是 **扩展贡献**，不是主线：

·         不强制输出 tree

·         允许 LLM：

o        把叶子扩展成小型 graph

o        或 rule-based substructure

·         最终：

o        Forest 变成 **Tree–Graph Hybrid Ensemble**

⚠️ 建议：

·         项目主线先不强调这个

·         放在：

o        Extension

o        Ablation

o        Future work

 

**七、为什么这套方法****“****合理****”****？（你刚才说****“****我不太懂为啥更高****”——****这里我帮你补）**

**核心原因只有一句话：**

**我们用** **LLM** **弥补了决策树在****“****跨特征关系建模****”****上的结构性弱点**

具体来说：

·         Random Forest 的强项：

o        variance reduction

o        decorrelation

·         但它假设：

o        base learner 足够“弱但合理”

·         你的方法：

o        **让每棵树稍微****“****聪明一点****”**

o        但仍保持随机性和多样性

·         结果：

o        Forest 不塌

o        Bias ↓

o        Variance 不明显 ↑

这就是为什么：

·         树数可以减少

·         性能还能提高

 

**八、实验设计（你图里那些数据集正好用得上）**

你已经说对了：

·         MNIST

·         USPS

·         Pendigits

·         Letter

·         Optical Recognition

·         Protein

·         SenseIT

·         Connect-4

这些正是：

**Tree / Rule-based / Ensemble** **方法的经典** **benchmark**

对比：

·         CART

·         TnT / RF

·         你们的 **LLM-augmented Trees**

指标：

·         Accuracy

·         #S（模型规模 / 节点数 / 规则数）

 

**九、项目里可以明确强调的两个创新点（你刚才也提到了）**

**⭐** **创新点** **1****：****LLM-based feature dependence discovery**

·         两种方法：

o        Tree-based local view

o        Global conditional view

·         可对比

·         可做 ablation

 

**⭐** **创新点** **2****：****LLM-enhanced base learner construction**

·         不改 ensemble 框架

·         只增强 base learner

·         计算复杂度友好

·         可扩展为 graph / rule hybrid



 

修好 conda / python / pip 环境

 

装好全部依赖

 

确认示例数据存在

 

跑通 run_randforest.py

 

成功得到 baseline 指标

Accuracy = 0.8986

 

AUC = 0.9131

 

LogLoss = 0.2269

 

F1 = 0.3445

成功生成规则/中间结果：

阶段1-2:

**bank**

- RF Accuracy = 0.8986
- RF AUC = 0.9131
- Fused Accuracy = 0.8993
- Fused AUC = 0.9153

**mnist**

- RF Accuracy = 0.6908
- RF AUC = 0.9419
- Fused Accuracy = 0.8277
- Fused AUC = 0.9743

数据集
 → 随机森林抽规则
 → 规则转 Prompt
 → LLM 生成新规则
 → 新规则转 Python
 → 误差修正训练
 → 和 RF 融合
 → 看 Accuracy / AUC

 

mnist + Prompt-A + 10 answers + n_ensemble=9 + eta=0.2

 



 

Relation-Aware Random Forest

我现在其实是想让决策树生成方式让LLM改进，之前原理是bootstrap随机抽取样本提取特征，弄三层 是不？现在是让LLM模型学习特征与特征之间的关系，最好是能生成一个特征之间的关系网图，然后让每一个决策树的特征更能代表整体，提高每一个base learner的性能。而不是仅仅让LLM混合原来的方法。

普通 fusion 是：

![img](data:image/png;base64,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)

也就是两个模型各做各的，最后融合。

而你现在的想法是：

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOUAAAAeCAMAAADkQin+AAAAAXNSR0IArs4c6QAAAN5QTFRFAAAAAAAAAAA6AABmADo6ADpmADqQAGaQAGa2OgAAOgA6OgBmOjoAOjo6OjpmOjqQOmaQOma2OpC2OpDbZgAAZgA6ZgBmZjoAZjqQZmZmZmaQZma2ZpC2ZpDbZra2ZrbbZrb/kDoAkDo6kDpmkGYAkGY6kGZmkGaQkJA6kJC2kLaQkLbbkLb/kNu2kNvbkNv/tmYAtmY6tmZmtpA6tpBmtpCQttuQttvbttv/tv/btv//25A625Bm25CQ27Zm27aQ29uQ2/+22////7Zm/7aQ/9uQ/9u2/9vb//+2///bHalIwwAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAADtUlEQVRYR+1XbVfTMBRuOhjUN8BNERW2qhMVdR2ojHUOWQg2//8Pee9N0qYkKSAcztHTfmi2vNx7n+e+pVHUPi0DLQMtAy0DLQMtAy0DmoGMdWZ+MuS3DWsl35zeI2d13X+p2DY5Ww+ATJ8u7RXxeNigjTPzBHddjBLGNqffa1Kvp5t2Ff1etVtOQNjKoBl/ZbJMrbPWIXdePBiHhfL1ZcTjcSSzEEqRYDScJqHQadR9GaVIutNokTTxjkdKk4tnftu5a03eDbvhHKQgykh89lMhEhUzxfNAgljHPLrrviz6aIp800C7EmdM5p2TEVt1NMuUjJIjBq6eoPkWMwGXEsrQk5nF+ZWm+XTXUZbCmiO2Mjlfez8ViRO1QkXDfJx3ZnMdGKHgNqoUyt+TjV/bQPU8Yav4X42Kff2YLYuExbuRTBnrFX14RZxROHh011AW/UAtcUAbk5GWoq9xVLtKv/D405aZLgtVVtYa2310BmxePVh0ZpMDiBeY0CNnZBiCYWt6S465BfMyQwbodYhB5dVtVx/OHHuDIaTVwuAJtXKq6FfhTIaEH30mw4wWUALh6ZlRowTXooW0RXGLLJPVObyKVyjdq9tGScehqjdliDZTmUxVCKLysu3llB2m10SJgk0BMSMURcUQISJByiP4xkQsdrbXI04+8uq2UMqUZAEtORCJR8wIgaTUALkqCZXJZIWnZVZRY8wrj+CPcMQaCJpnIwf0EwIvSsTFh/CigLUi1tLtokSPGvGluZlSg7lioUSstXarnWqOFS8hx/Scjw2zZFlH9IkkhsQUu2YEdEpO5UvVW4hmkQz2lyLZo4AtTa/prkUs9RHsAqYhlo3x8AOuFTumP5IDaSskyOlrmcZfJ6x7ts2o2Og6J9/O4NdCucHHho1yHtM+FSTkbmDUjNDDY7iqyEkZsRBng+iCLiUyXYGjWax6rU93TT1nUJmPMCZdlD8wBc+/aJTKZHpDNG0tfx6nT8Z4P1GNVPUsdDyuKiyNlx9VPrtL7AvoHnmkOokZo+hi9Ahwbx5T6yDHQpNB5KjIzmWf7jrJeHDvHRDjopyB9+T+mUbpmIwlD8MnV2l7w7uP7dNb/77y7lNq8KCEKT40NzrnusaBzhx40NemG95jb43MFuC/fvgSxoMSDh/ONEo3+hBhOozEw5OPVIzljb5J7hSlozsonYpk8WJpRrxWcEhYhdL9jEL+cA1uAfpD8h/4vsQ+iXcPM0LGx2OIRryL9KL7/SS+Wz+30loG/ncG/gBKqqA9WPMeKwAAAABJRU5ErkJggg==)

其中 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAeCAMAAABpA6zvAAAAAXNSR0IArs4c6QAAAIFQTFRFAAAAAAAAAAA6AABmADo6ADqQAGa2OgAAOgA6OpC2OpDbZgAAZgA6ZjqQZmZmZpDbZra2ZrbbZrb/kDoAkDo6kGYAkGaQkJA6kNu2kNv/tmYAtmY6tmZmtpA6ttuQttv/tv//25A625Bm27Zm2/+22////7Zm/7aQ/9uQ//+2///bcnYe9AAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAz0lEQVQ4T+2SyxKCMAxFExSq4JP6rKigUmn//wNNX266qEudMZs7kxwuyQWAf31xAvrEEMfr5IYDy69wZ3UKVFXeA+jNMQWKLIlYC1UVKSs3l5hczoFi1BoaUwtobk6BDusO0Zp71RztBAaGJYkHja/0pkGFW6rzbWHD4XTRMHHnB232ZqQWM9eWuAR9Nu4ReDFPPg8ehBvDbLWlt0RgKwrQu0cA3ynGIHVkrT4ANS+bNgbttWreB20oCNo9Ak1+VKXXKadvQL8KxWhz/IF6AWQrEd7CHFzbAAAAAElFTkSuQmCC)是 LLM 生成的特征关系图。

也就是说：

**LLM** **不再是最终预测器，而是树生成过程的先验结构提供者。**

目标：

 

所以现在我的代码暂时是没有这个改进的是吧？其实可以先不把tree-graph，先让LLM识别特征之间的关系，然后在原来RF的每一棵树（3层）然后ensemble100棵，在最末端的节点延伸加一个节点（变为4层），然后ensemble减半50棵，这样看看会不会在复杂度不增大太多的情况下提高准确率呢？

**LLM** **直接参与** **RF** **每一棵树的生成机制**。

也就是说，目前更像：

![img](data:image/png;base64,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)

而不是：

![img](data:image/png;base64,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)

**baseline**

- RF
- max depth = 3
- n_trees = 100

**improved version**

- RF front structure unchanged
- at terminal or near-terminal nodes, use     LLM-inferred feature relations to select one extra split feature
- max depth = 4
- n_trees = 50

核心思想：

在不显著增加整体复杂度的前提下，用 LLM 引导的局部结构扩展提升单个 base learner 的判别能力。

**实验** **1****：纯** **baseline**

- depth = 3
- n_trees = 100

**实验** **2****：只加深，不用** **LLM**

- depth = 4
- n_trees = 50

这组非常重要。
 因为你要先知道：

提升到底是因为“树更深了”，还是因为“LLM 指导有用”。

**实验** **3****：加深** **+ LLM-guided leaf expansion**

- depth = 4
- n_trees = 50
- 第4层分裂特征由 LLM 关系引导

如果实验 2 和实验 3 比：

- 2 没怎么提升
- 3 提升明显

那你就能说：

**不是单纯加深树，而是** **LLM** **识别的** **feature relations** **真有效。**

这个是论文里最关键的识别逻辑。

100棵 × 3层 baseline

50棵 × 4层 no-LLM

50棵 × 4层 with LLM-guided expansion

 

**算法流程**

**1.** **输入**

- 训练集 ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAMAAAAM7l6QAAAAAXNSR0IArs4c6QAAAG9QTFRFAAAAAAAAAAA6AABmADpmADqQAGa2OgAAOgA6OgBmOjqQOpC2OpDbZgAAZjoAZmZmZpDbZrbbZrb/kDoAkDpmkNv/tmYAtmY6tmZmtpBmtrZmtv//25A625Bm27Zm2////7Zm/9uQ/9u2//+2///bxtQLSAAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAAZdEVYdFNvZnR3YXJlAE1pY3Jvc29mdCBPZmZpY2V/7TVxAAAAvUlEQVQ4T+2RyxaCIBCGB8noolmRWoEJ6fs/YyO3gx7ctGjVv4KZ+WZ+BoC/vtrAixEqRk5IkcabrAZQ5Nin03qH6Wa7NnrkBchDxHZsIwDHOULmzzirz5pdAYbSmdGMilnrkSOo9y4ofZtQ0+Q9KAcr+ljQIPGpNwuj8TDG44oKbeGhRBtTs1iI3A1sXICcNuPOlqkQQnGC4PtE8KVR2kILmTlG3tesoL34a+frEptua7nyedZsFrok4J+FPoBTCwR7Z7mCAAAAAElFTkSuQmCC)
- RF 参数：树数、最大深度、特征采样方式
- LLM 输出的特征关系映射     feature_relation_map

**2.** **训练基准随机森林前三层**

对每棵树：

- bootstrap 采样
- 递归分裂到 depth = 3

**3.** **叶节点筛选**

对每棵树的每个叶节点：

- 若样本数不足，跳过
- 若 impurity 很低，跳过
- 否则进入扩展阶段

**4. LLM-guided** **节点扩展**

对候选叶节点：

- 找到最后一个分裂特征 f_last
- 从 feature_relation_map[f_last] 中取     top-k 特征
- 用这些特征在当前叶节点数据上计算最优分裂
- 新增一个子节点

**5.** **输出模型**

得到一组扩展后的树，形成新的 forest

 

**七、伪代码**

你论文里可以先写成这种风格：

Algorithm 1: LLM-Guided Leaf Expansion Random Forest
 
 Input:
   Training data (X, y)
   Number of trees B
   Base depth d = 3
   Expanded depth d' = 4
   Feature relation map R from LLM
   Leaf expansion thresholds: min_samples_leaf, impurity_threshold
 
 For b = 1 to B:

1. Draw a bootstrap sample (X_b, y_b)
2. Train a decision tree T_b up to depth d using standard RF splitting
3. For each leaf node l in T_b:

​      a. If size(l) < min_samples_leaf: continue
​      b. If impurity(l) < impurity_threshold: continue
​      c. Let f_last be the last split feature on the path to l
​      d. Retrieve candidate feature set C = R[f_last]
​      e. Search the best split only within C on samples in l
​      f. If a valid split exists, expand l by one additional node
 Return:
   Expanded forest {T_1, ..., T_B}

 

 

# 实验结果：

# V1

叶节点加一层，LLM-guided feature candidates

- **relation_map** **并不行**
- 而且现在这版 **leaf expansion** **仍然明显不如** **Deeper_RF_50xDepth4 = 0.8043**
- 所以目前可以下一个很清楚的结论：

**“****在树末端补一层****”** **这条** **V1** **路线，至少按现在这个实现方式，并不能有效增强** **base learner****。**

Scaling features to [0, 1]...

Train/test split...

Train shape: (60000, 784)

Test shape : (10000, 784)

 

[Experiment 1] Baseline RF: 100 trees x depth 3

Training time = 10.75 sec

 

Baseline_RF_100xDepth3

\----------------------

Accuracy = 0.7473

AUC   = 0.9649310995

EvalTime = 0.29 sec

 

Top 10 important features from baseline:

[409, 461, 350, 155, 437, 433, 373, 569, 434, 154]

 

Sample relation_map entries:

feature 409 -> [437, 381, 350]

feature 461 -> [460, 462, 489]

feature 350 -> [378, 405, 406]

feature 155 -> [154, 597, 569]

feature 437 -> [409, 381, 350]

 

[Experiment 2] Deeper RF: 50 trees x depth 4

Training time = 6.28 sec

 

Deeper_RF_50xDepth4

\-------------------

Accuracy = 0.8043

AUC   = 0.9733715246

EvalTime = 0.15 sec

 

[Experiment 3] LLM-guided Leaf Expansion Forest

Training time = 6.54 sec

 

LeafExpansion_RF_50Trees_BaseDepth3_Plus1

\-----------------------------------------

Accuracy = 0.7445

AUC   = 0.9632121142

EvalTime = 8.14 sec

 

Leaf Expansion Statistics

\-------------------------

Candidate leaves checked : 399

Leaves actually expanded : 100

 

Summary

\-------

Baseline_RF_100xDepth3: Accuracy=0.7473, AUC=0.9649310995, EvalTime=0.29s

Deeper_RF_50xDepth4: Accuracy=0.8043, AUC=0.9733715246, EvalTime=0.15s

LeafExpansion_RF_50Trees_BaseDepth3_Plus1: Accuracy=0.7445, AUC=0.9632121142, EvalTime=8.14s

 

# V2

不只是最后一层，倒数两层都加 guidance

Relation-Subspace Forest

这个版本的核心不是“在叶节点补一层”，而是：

- 每棵树训练前先选一个 **anchor feature**
- 根据 relation_map 找到和它相关的一组特征
- 再加少量随机补充特征
- 用这个“结构化特征子空间”训练一棵 depth=4 的树

这样更接近你最开始想要的：
 **让关系结构参与** **base learner** **的生成，而不是事后修补。**

**1. Baseline_RF_100xDepth3**

这是你最开始的浅层 baseline。

**2. Deeper_RF_50xDepth4**

这是你现在已知最强的简单 baseline。
 它是你后面真正要打的基准线。

**3. TopFeatureSubspace_RF_50xDepth4**

这个很重要。
 它检验的是：

如果我只是把树限制在 top features 上训练，会不会已经变好？

如果它已经很好，说明“重要特征筛选”本身就有用。

**4. RelationSubspace_RF_50xDepth4**

这是你的核心方法。
 它检验的是：

在 top features 基础上，再引入 relation-guided 子空间，能不能比纯 top-feature 子空间更进一步。

| **模型**               | **Accuracy** |
| ---------------------- | ------------ |
| Baseline_RF_100xDepth3 | **0.7626**   |
| Deeper_RF_50xDepth4    | **0.8161**   |
| TopFeatureSubspace_RF  | **0.6644**   |
| RelationSubspace_RF    | **0.6696**   |

 

真正有效的改进不在“feature subset”，而在“split decision”。应该让LLM 参与 split feature selection

 

Instead of modifying the ensemble aggregation or restricting feature subsets, we integrate LLM guidance into the feature selection stage of tree induction. Specifically, given a candidate set of features at each node, the LLM ranks features based on inferred relationships and contextual relevance, and the tree only evaluates the top-ranked features using traditional impurity criteria.

| **dataset**    | **samples** | **features** | **classes** |
| -------------- | ----------- | ------------ | ----------- |
| MNIST          | 70000       | 784          | 10          |
| USPS           | 9298        | 256          | 10          |
| Pendigits      | 10992       | 16           | 10          |
| Letter         | 20000       | 16           | 26          |
| Connect-4      | 67557       | 42           | 3           |
| Optical digits | 5620        | 64           | 10          |
| Protein        | 45730       | 9            | 3           |
| SenseIT        | 78823       | 50           | 3           |

 

 

V3

从 feature relation list 升级到 feature graph

 

V4

从 graph-guided tree 走向 graph-based learner

 



 

 

把 DeLTa 这个项目升级成真正论文级算法：

LLM-Guided Decision Graph

 