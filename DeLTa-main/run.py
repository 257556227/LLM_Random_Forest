""" 
If you want to download TabPFN to your local machine for regression tasks, 
please download the TabPFN weights to the "DeLTa-main/model/models/models_diff/" directory 
and set the environment variable with the command: 
export TABPFN_MODEL_CACHE_DIR="DeLTa-main/model/models/models_diff/"   #It is recommended to use an absolute path.
"""


import os
import subprocess
import argparse
from model.utils import resolve_small_model_for_runtime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run DeLTa training jobs')
    parser.add_argument(
        '--num_answers',
        type=int,
        default=int(os.environ.get('DELTA_NUM_ANSWERS', '10')),
        help='Number of answer trees to train (default from DELTA_NUM_ANSWERS or 10)',
    )
    return parser.parse_args()


def start(args):
    # Dataset-specific configuration
    from dataset_config import dataset_params, default_params

    # 支持通过环境变量指定数据集，例如: DELTA_DATASETS=mnist,bank
    dataset_env = os.environ.get('DELTA_DATASETS')
    if dataset_env:
        datasets = [d.strip() for d in dataset_env.split(',') if d.strip()]
    else:
        datasets = ['bank']
    
    # Common parameters
    shot_range = ['full']
    n_answers = args.num_answers
    model_type = 'DeLTa'
    
    for dataset in datasets:
        # Get dataset-specific parameters or use defaults
        params = dataset_params.get(dataset, default_params)
        mds = params['mds']
        mls = params['mls']
        n_estimators = params['n_estimators']
        # Directly assign knn from parameters instead of looping
        requested_small_model = params['small_model']
        small_model = resolve_small_model_for_runtime(requested_small_model)
        if small_model != requested_small_model:
            print(
                f"Runtime fallback for {dataset}: small_model {requested_small_model} -> {small_model} "
                "because CUDA TabPFN is unavailable in the current environment."
            )
        
        # Create results directory for dataset if it doesn't exist
        results_dir = f"results/{dataset}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")
        
        # Build base classify rule path
        base_classify_rule = f'model.llm_rule.{dataset}.'
        
        # Iterate through all parameter combinations 
        for tree in n_estimators:
            for md in mds:
                for ml in mls:
                    # Generate classification rules inside the loop to use current md and ml values
                    classify_rules = []
                    for i in range(len(shot_range)):
                        current_rules = []
                        for j in range(n_answers):
                            rule = f'tree_{shot_range[i]}_md{md}_ml{ml}_{j}'
                            current_rules.append(rule)
                        classify_rules.append(current_rules)
                    
                    # Generate commands for each shot range and answer
                    cmds = []
                    for i in range(len(shot_range)):
                        for j in range(n_answers):
                            cmd_parts = [
                                'python', 'train.py',
                                '--task_type', 'full',
                                '--small_model', small_model,
                                '--classify_rule', f'{base_classify_rule}{classify_rules[i][j]}',
                                '--dataset', dataset,
                                '--dataset_path', 'example_datasets',
                                '--seed_num', '1',
                                '--model_type', model_type,
                                '--gpu', '0',
                                '--cat_policy', 'ordinal',
                                '--save_npy',
                                f'results/{dataset}/RF_md{md}_ml{ml}_tree{tree}_full_{small_model}_{j}'
                            ]
                            
                            log_file = f"results/{dataset}/RF_md{md}_ml{ml}_tree{tree}_{dataset}_shotfull_{j}_{small_model}.log"
                            cmds.append((cmd_parts, log_file))
                    
                    # Execute all commands for current parameter combination
                    for cmd_parts, log_file in cmds:
                        try:
                            print(f"Executing command: {' '.join(cmd_parts)} > {log_file}")
                            with open(log_file, 'w', encoding='utf-8') as f:
                                subprocess.run(cmd_parts, check=True, stdout=f, stderr=subprocess.STDOUT)
                        except subprocess.CalledProcessError as e:
                            print(f"Command failed: {' '.join(cmd_parts)}\nError: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    start(args)
