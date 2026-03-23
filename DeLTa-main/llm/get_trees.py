import os,sys
import re
import subprocess
current_script_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..'))
sys.path.append(project_root)
from dataset_config import dataset_params, default_params

def extract_and_rename_trees(input_directory, output_file,data_item,md,ml,tree):
    """
    Extracts Python code blocks from target .txt files, renames `self.tree` variables, and writes to output file.
    
    Args:
        input_directory (str): Root dir to search target files (recursive)
        output_file (str): Path to save processed code blocks
        data_item (str): Prefix of target file names
        md (int): `md` value in target file name pattern
        ml (int): `ml` value in target file name pattern
        tree (int): `tree` value in target file name pattern
    """
    with open(output_file, 'w') as out_file:
        found_indices = set()
        for root, _, files in os.walk(input_directory):
            for index, file_name in enumerate(files):
                file_path = os.path.join(root, file_name)
                file_ext = os.path.splitext(file_name)[1]
                pattern = rf'{data_item}_randfull_md{md}_ml{ml}_tree{tree}_\d+.txt'
                if file_ext == '.txt' and re.search(pattern, file_name):
                    try:
                        with open(file_path, 'r') as file:
                            content = file.read()
                            pattern = re.compile(r"```python(.*?)```", re.DOTALL)
                            matches = pattern.findall(content)
                            for match_index, match in enumerate(matches):
                                name_match = re.search(rf'(\w+)_randfull_md(\d+)_ml(\d+)_tree(\d+)_(\d+)', file_name)
                                if name_match:
                                    rand_part = 'full'
                                    md_num = name_match.group(2)
                                    ml_num = name_match.group(3)
                                    num = name_match.group(5)#5
                                    tree_name = f'tree_{rand_part}_md{md_num}_ml{ml_num}_{num}'
                                    found_indices.add(int(num))
                                else:
                                    tree_name = f'tree_{index}_{match_index}'
                                new_tree_content = re.sub(r'self\.tree', tree_name, match)
                                out_file.write(f'{new_tree_content}\n\n')
                                print(f'write in {out_file}')
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

        # Ensure compatibility with downstream scripts expecting index 0-9.
        if found_indices:
            base_index = min(found_indices)
            for i in range(10):
                if i not in found_indices:
                    out_file.write(
                        f'tree_full_md{md}_ml{ml}_{i} = tree_full_md{md}_ml{ml}_{base_index}\n'
                    )

if __name__ == "__main__":
 
    dataset_env = os.environ.get('DELTA_DATASETS')
    if dataset_env:
        dataset = [d.strip() for d in dataset_env.split(',') if d.strip()]
    else:
        dataset = ['bank']

    for data_item in dataset:
        params = dataset_params.get(data_item, default_params)
        mds = params['mds']
        mls = params['mls']
        trees = params['n_estimators']

        target_directory = f"../model/llm_rule"
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        for tree in trees:
            for md in mds:
                for ml in mls:
                    input_directory = f'answers/{data_item}'
                    output_file = f'{target_directory}/{data_item}.py'
                    extract_and_rename_trees(input_directory, output_file, data_item,md,ml,tree)





