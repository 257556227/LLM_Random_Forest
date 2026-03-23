"""
Comprehensive grid search with prompt engineering variations
Strategies:
1. Prompt styles: simple, detailed, chain-of-thought, cot-fewshot
2. Temperature: 0, 0.3, 0.7
3. Max tokens: 256, 512, 1024
4. Different few-shot examples
5. Different ranking strategies
"""

import subprocess

# Prompt engineering variations
prompt_styles = [
    "simple",      # Simple ranking
    "detailed",    # Detailed analysis
    "cot",         # Chain-of-thought
    "cot_fs",      # Chain-of-thought + few-shot
]

# Datasets to optimize
datasets = ["house_16H_reg", "california_housing"]

# Temperature settings
temperatures = ["0", "0.3"]

results = []

for ds in datasets:
    for style in prompt_styles:
        for temp in temperatures:
            cmd = f"""python run_leaf_expansion_mnist.py --dataset {ds} --mode llm_guided --relation_source mock_llm --base_max_depth 3 --top_k_leaves 10 --prompt_style {style} --temperature {temp} --save_summary"""
            print(f"Testing: {ds} style={style} temp={temp}")
            subprocess.run(cmd, shell=True)
            results.append((ds, style, temp))

print(f"Total experiments: {len(results)}")
