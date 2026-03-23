import time
import shlex
import subprocess
import itertools
import argparse
import os,sys
import subprocess

# Get project root path and add to system path for module import
current_script_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..', '..'))
sys.path.append(project_root)
from dataset_config import dataset_params, default_params


def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch run get_answer with configurable load control")
    parser.add_argument('--num_queries', type=int, default=10, help='Queries per prompt chunk')
    parser.add_argument('--max_retries', type=int, default=5, help='Max retries per query')
    parser.add_argument('--retry_delay', type=float, default=5.0, help='Retry delay in seconds')
    parser.add_argument('--request_interval', type=float, default=0.0, help='Sleep seconds between successful requests')
    parser.add_argument('--split_rules_max_chars', type=int, default=0, help='Max chars per rules chunk, 0 means no split')
    parser.add_argument('--split_rules_overlap_chars', type=int, default=0, help='Overlap chars between adjacent chunks')
    parser.add_argument('--max_completion_tokens', type=int, default=0, help='Max completion tokens, 0 means model default')
    parser.add_argument('--normalize_tree_output', type=int, default=1, help='Normalize answer to strict self.tree python code block')
    return parser.parse_args()

def start(args):
    """
    Generate and execute commands to run get_answer.py for multiple dataset parameters
    
    Iterates over datasets, mds, mls, and tree numbers from config,
    creates output directories, builds commands, and executes them sequentially.
    """
    # Define target datasets to process
    dataset_env = os.environ.get('DELTA_DATASETS')
    if dataset_env:
        datasets = [d.strip() for d in dataset_env.split(',') if d.strip()]
    else:
        datasets = ['bank']
    cmds = []
    
    # Generate commands for each parameter combination
    for dataset in datasets:
        # Get dataset-specific parameters (fallback to default if not found)
        params = dataset_params.get(dataset, default_params)
        mds = params['mds']
        mls = params['mls']
        trees = params['n_estimators']
        
        # Create output directory for answers if not exists
        target_answer_path = f"../answers/{dataset}"
        if not os.path.exists(target_answer_path):
            print(f"Creating prompts directory: {target_answer_path}")
            os.makedirs(target_answer_path)
        
        # Build command for each (tree, md, ml) combination
        for tree in trees:
            for md in mds:
                for ml in mls:
                    log_file = f"{dataset}_randfull_md{md}_ml{ml}_tree{tree}.txt"
                    cmd_parts = [
                        'python', 'get_answer.py',
                        '--base_rule_path', f"../prompts/{dataset}",
                        '--rule_path', log_file,
                        '--target_answer_path', target_answer_path,
                        '--num_queries', str(args.num_queries),
                        '--max_retries', str(args.max_retries),
                        '--retry_delay', str(args.retry_delay),
                        '--request_interval', str(args.request_interval),
                        '--split_rules_max_chars', str(args.split_rules_max_chars),
                        '--split_rules_overlap_chars', str(args.split_rules_overlap_chars),
                        '--max_completion_tokens', str(args.max_completion_tokens),
                        '--normalize_tree_output', str(args.normalize_tree_output),
                        ]
                    cmds.append(cmd_parts)
    
    # Execute generated commands sequentially
    for cmd in cmds:
        try:
            print(f"Executing command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error: {e}")


if __name__ == "__main__":
    # Entry point: start command generation and execution
    args = parse_arguments()
    start(args)
