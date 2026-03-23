import os
import time
import argparse
import ast
import importlib.util
import re
import pprint
from pathlib import Path
from openai import APIConnectionError, APIError, BadRequestError, OpenAI, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_openai_settings_helper():
    helper_path = PROJECT_ROOT / "llm" / "openai_local_config.py"
    spec = importlib.util.spec_from_file_location("delta_openai_local_config", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OpenAI config helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_openai_settings

get_openai_settings = _load_openai_settings_helper()

def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenAI API query script")
    parser.add_argument('--base_rule_path', type=str,help="Base path for prompt files")
    parser.add_argument('--rule_path', type=str,help="Path to the prompt file")
    parser.add_argument('--target_answer_path', type=str,help="Path to save the results")
    parser.add_argument('--num_queries', type=int, default=10, help="Queries per prompt chunk")
    parser.add_argument('--max_retries', type=int, default=5, help="Max retries for each query")
    parser.add_argument('--retry_delay', type=float, default=5.0, help="Retry delay in seconds")
    parser.add_argument('--request_interval', type=float, default=0.0, help="Sleep seconds between successful requests")
    parser.add_argument('--split_rules_max_chars', type=int, default=0, help="Max chars per rules chunk, 0 means no split")
    parser.add_argument('--split_rules_overlap_chars', type=int, default=0, help="Overlap chars between adjacent chunks")
    parser.add_argument('--max_completion_tokens', type=int, default=0, help="Max completion tokens, 0 means model default")
    parser.add_argument('--normalize_tree_output', type=int, default=1, help="Normalize model output to strict self.tree code block")
    return parser.parse_args()


def _split_text_with_overlap(text, chunk_size, overlap):
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks


def build_chunked_prompts(prompt_content, split_rules_max_chars=0, split_rules_overlap_chars=0):
    """
    Split the CART rules section into multiple prompt chunks while keeping
    meta information and output-format constraints intact.
    """
    if split_rules_max_chars <= 0:
        return [prompt_content]

    start_marker = "## CART tree rules"
    end_marker = "## CART tree rules end"

    start_idx = prompt_content.find(start_marker)
    end_idx = prompt_content.find(end_marker)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # Fallback: split whole prompt when markers are missing.
        return _split_text_with_overlap(
            prompt_content,
            chunk_size=split_rules_max_chars,
            overlap=split_rules_overlap_chars,
        )

    header = prompt_content[: start_idx + len(start_marker)]
    rules_body = prompt_content[start_idx + len(start_marker): end_idx]
    tail = prompt_content[end_idx:]

    body_chunks = _split_text_with_overlap(
        rules_body,
        chunk_size=split_rules_max_chars,
        overlap=split_rules_overlap_chars,
    )

    return [f"{header}{chunk}{tail}" for chunk in body_chunks]


def _extract_tree_literal(text):
    """Extract the dict literal part from 'self.tree = {...}' with brace matching."""
    marker_match = re.search(r'self\.tree\s*=\s*\{', text)
    if not marker_match:
        return None
    start = marker_match.start()
    brace_start = text.find('{', marker_match.start())
    if brace_start == -1:
        return None

    depth = 0
    end = None
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None

    assignment_text = text[start:end + 1]
    return assignment_text


def _assign_missing_leaf_ids(node, leaf_counter):
    if not isinstance(node, dict):
        return {"id": f"leaf_{leaf_counter[0]}"}

    if "id" in node:
        return node

    # Internal node: repair required keys
    if "left" not in node:
        node["left"] = {"id": f"leaf_{leaf_counter[0]}"}
        leaf_counter[0] += 1
    if "right" not in node:
        node["right"] = {"id": f"leaf_{leaf_counter[0]}"}
        leaf_counter[0] += 1

    node["left"] = _assign_missing_leaf_ids(node["left"], leaf_counter)
    node["right"] = _assign_missing_leaf_ids(node["right"], leaf_counter)
    return node


def normalize_tree_answer(text):
    """
    Convert free-form LLM text into a strict python code block:
    ```python
    self.tree = {...}
    ```
    Returns original text if extraction/parsing fails.
    """
    candidate = _extract_tree_literal(text)
    if not candidate:
        return text

    # Remove trailing inline comments to improve parsing robustness
    candidate_no_comment = re.sub(r'#.*', '', candidate)

    try:
        parsed = ast.parse(candidate_no_comment)
        tree_value = None
        for stmt in parsed.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and target.attr == 'tree':
                        tree_value = ast.literal_eval(stmt.value)
                        break
                if tree_value is not None:
                    break

        if not isinstance(tree_value, dict):
            return text

        repaired_tree = _assign_missing_leaf_ids(tree_value, leaf_counter=[1])
        formatted = pprint.pformat(repaired_tree, width=100, sort_dicts=False)
        return f"```python\nself.tree = {formatted}\n```"
    except (SyntaxError, ValueError, TypeError):
        return text

def query_openai(
    file_path,
    num_queries=10,
    max_retries=5,
    retry_delay=5,
    request_interval=0.0,
    split_rules_max_chars=0,
    split_rules_overlap_chars=0,
    max_completion_tokens=0,
    normalize_tree_output=1,
):
    """
    Query OpenAI API with content from a prompt file
    
    Parameters:
        file_path: Path to the file containing prompt content
        num_queries: Number of queries to perform
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay between retries in seconds
    
    Returns:
        List of query results
    """
    openai_settings = get_openai_settings()
    api_key = openai_settings["api_key"]
    base_url = openai_settings["base_url"]
    model = openai_settings["model"]

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required. You can set it via environment variables or DeLTa-main/local_openai_config.json"
        )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_content = file.read().strip()

    prompt_chunks = build_chunked_prompts(
        prompt_content,
        split_rules_max_chars=split_rules_max_chars,
        split_rules_overlap_chars=split_rules_overlap_chars,
    )

    results = []
    
    for i in range(num_queries):
        retries = 0
        while retries < max_retries:
            try:
                prompt_for_query = prompt_chunks[i % len(prompt_chunks)]
                
                # Call OpenAI API
                request_kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_for_query}],
                }
                if max_completion_tokens > 0:
                    request_kwargs["max_completion_tokens"] = max_completion_tokens

                response = client.chat.completions.create(**request_kwargs)

                answer_text = response.choices[0].message.content
                if normalize_tree_output:
                    answer_text = normalize_tree_answer(answer_text)

                results.append(answer_text)

                if request_interval > 0:
                    time.sleep(request_interval)
                break
                
            except (APIConnectionError, APIError, BadRequestError, RateLimitError, OSError, ValueError) as e:
                print(f"Error: {e}, retrying {retries + 1}/{max_retries}...")
                retries += 1
                time.sleep(retry_delay)
        
        if retries == max_retries:
            print(f"Query {i + 1} exceeded maximum retries, skipping...")
    
    return results

def main():
    args = parse_arguments()
    
    # Construct input file path
    input_file = os.path.join(args.base_rule_path, args.rule_path)
    
    # Execute query (default: 10 query)
    results = query_openai(
        input_file,
        num_queries=args.num_queries,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        split_rules_max_chars=args.split_rules_max_chars,
        split_rules_overlap_chars=args.split_rules_overlap_chars,
        max_completion_tokens=args.max_completion_tokens,
        normalize_tree_output=args.normalize_tree_output,
    )
    
    # Save results
    for idx, result in enumerate(results):
        # Construct output filename
        file_name = f"{os.path.splitext(args.rule_path)[0]}_{idx}.txt"
        output_file = os.path.join(args.target_answer_path, file_name)
        
        # Write result to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        print(f"Result {idx} saved to: {output_file}")

if __name__ == "__main__":
    main()

    
