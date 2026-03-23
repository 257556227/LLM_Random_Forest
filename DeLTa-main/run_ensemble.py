import os
import subprocess
import argparse
from model.utils import resolve_small_model_for_runtime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run DeLTa ensemble jobs')
    parser.add_argument(
        '--n_ensemble',
        type=int,
        default=int(os.environ.get('DELTA_N_ENSEMBLE', '9')),
        help='Max ensemble index (default from DELTA_N_ENSEMBLE or 9)',
    )
    parser.add_argument(
        '--eta',
        type=float,
        default=None,
        help='Optional fusion eta override; if omitted use dataset_config eta',
    )
    return parser.parse_args()


def start(args):
    """Generate and execute ensemble.py commands for multiple parameter combinations"""
    # Dataset-specific configuration
    from dataset_config import dataset_params, default_params
    
    # Common parameters
    n_ensembles = [str(args.n_ensemble)]
    # 支持通过环境变量指定数据集，例如: DELTA_DATASETS=mnist,bank
    dataset_env = os.environ.get('DELTA_DATASETS')
    if dataset_env:
        datasets = [d.strip() for d in dataset_env.split(',') if d.strip()]
    else:
        datasets = ['bank']
    
    # Iterate through target datasets
    for dataset in datasets:
        # Get dataset-specific parameters or use defaults
        params = dataset_params.get(dataset, default_params)
        mds = params['mds']
        mls = params['mls']
        n_estimators = params['n_estimators']
        num_class = params['num_class']
        requested_small_model = params['small_model']
        small_model = resolve_small_model_for_runtime(requested_small_model)
        if small_model != requested_small_model:
            print(
                f"Runtime fallback for {dataset}: small_model {requested_small_model} -> {small_model} "
                "because CUDA TabPFN is unavailable in the current environment."
            )
        
        # Iterate through all parameter combinations for this dataset
        for md in mds:
            for ml in mls:
                for tree in n_estimators:
                    for n_ensemble in n_ensembles:
                        # Build command to execute ensemble.py with current parameters
                        cmd_parts = [
                            'python', 'ensemble.py',
                            '--small_model', small_model,
                            '--num_classes', str(num_class),
                            '--n_ensemble', n_ensemble,
                            '--md', str(md),
                            '--ml', str(ml),
                            '--tree', str(tree),
                            '--dataset', dataset,
                        ]
                        if args.eta is not None:
                            cmd_parts.extend(['--eta_override', str(args.eta)])

                        # Define log file path for command output
                        log_file = f'results/{dataset}/e_RF_md{md}_ml{ml}_tree{tree}_{dataset}_{small_model}_{n_ensemble}.log'  
                        
                        # Execute command and redirect output to log file
                        try:
                            print(f"Executing command: {' '.join(cmd_parts)} > {log_file}")
                            with open(log_file, 'w', encoding='utf-8') as f:
                                subprocess.run(cmd_parts, check=True, stdout=f, stderr=subprocess.STDOUT)
                        except subprocess.CalledProcessError as e:
                            print(f"Command failed: {' '.join(cmd_parts)}\nError: {e}")

if __name__ == "__main__":
    # Entry point: start command generation and execution
    args = parse_arguments()
    start(args)
