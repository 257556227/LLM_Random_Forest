"""
Systematic grid search for all remaining datasets
Strategies to test:
1. base_max_depth: 2, 3, 4, 5
2. top_k_leaves: 3, 5, 7, 10
3. leaf_selection_strategy: impurity_mass, sample_count, variance_mass
"""

import subprocess
import itertools

datasets = ["house_16H_reg", "california_housing", "jannis", "credit-g"]
depths = [2, 3, 4, 5]
leaves = [3, 5, 7, 10]
strategies = ["impurity_mass", "sample_count"]

results = []

for ds in datasets:
    for depth in depths:
        for n_leaves in leaves:
            for strategy in strategies:
                cmd = f"python run_leaf_expansion_mnist.py --dataset {ds} --mode llm_guided --relation_source mock_llm --base_max_depth {depth} --top_k_leaves {n_leaves} --leaf_selection_strategy {strategy} --save_summary"
                print(f"Testing: {ds} depth={depth} leaves={n_leaves} strategy={strategy}")
                subprocess.run(cmd, shell=True)
                results.append((ds, depth, n_leaves, strategy))

print(f"Total experiments: {len(results)}")
