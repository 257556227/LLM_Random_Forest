You are an expert in tabular machine learning domain. I will provide the meta information and CART tree rules.
Your goal is to produce ONE improved inference rule tree for better multiclass performance on MNIST.

## Meta information about dataset.
{
    "name": "mnist",
    "task_type": "multiclass",
    "num_classes": 10,
    "n_num_features": 784,
    "n_cat_features": 0,
    "train_size": 50000,
    "val_size": 10000,
    "test_size": 10000
}

## CART tree rules

## CART tree rules end

Requirements:
1) Learn the evolving process from the provided CART rules, do not copy directly.
2) Return ONLY one Python code block.
3) Output must be strictly:
```python
self.tree = { ... }
```
4) Use binary splits with keys: "feature", "threshold", "operator", "left", "right".
5) Every non-leaf node must include BOTH "left" and "right".
6) Every leaf must be exactly {"id": "leaf_k"} and each leaf id appears only once.
7) Limit leaf count to <= 20.
8) Keep thresholds concise (2 decimals preferred).

Do not include explanations, comments, markdown headings, or any extra text outside the single python code block.
