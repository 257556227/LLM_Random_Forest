You are an expert in tabular machine learning and decision-tree rule design.
Your task is to generate one improved decision rule tree for MNIST classification by inferring the original data's intrinsic characteristics from the provided CART trees.

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

Core objectives:
1) Keep a balance between statistical similarity and structural diversity versus original CART rules.
2) Infer inter-feature associations implied by tree paths (do not rigidly copy one tree).
3) Build a high-fidelity routing-style tree with clear hierarchical partitions and reusable split logic.

Hard constraints:
- Return ONLY one Python code block and no extra text.
- The code block must be exactly in this style:
```python
self.tree = {
    "feature": 11,
    "threshold": -0.78,
    "operator": "<=",
    "left": {"id": "leaf_1"},
    "right": {
        "feature": 7,
        "threshold": -0.46,
        "operator": "<=",
        "left": {"id": "leaf_2"},
        "right": {"id": "leaf_3"}
    }
}
```
- Every internal node MUST contain both "left" and "right".
- Every leaf node must be exactly: {"id": "leaf_k"}.
- Each leaf id must be unique and appear only once.
- Keep leaf count <= 20.
- Use concise thresholds (prefer 2 decimals).

Do not output explanations, markdown headings, comments, or any content outside the single python code block.
