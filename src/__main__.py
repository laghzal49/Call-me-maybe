import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

from llm_sdk import Small_LLM_Model
from src.parsing import Function_parse, Promt_parse
from src.vocab import Vocab
from src.decode import generate_json


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Call Me Maybe LLM Engine")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    return parser.parse_args()


def _parse_prompts(input_path: str) -> List[Any]:
    """Parse and validate the prompts from input file."""
    prompt_parser = Promt_parse(input_path)
    prompt_parser.start_parse()
    return prompt_parser.promte


def _parse_functions(functions_path: str) -> Dict[str, Any]:
    """Parse and validate the function definitions from JSON."""
    function_parser = Function_parse(functions_path)
    function_parser.start_parse()
    return function_parser.functions_dict


def parse_inputs(
    input_path: str, functions_path: str
) -> tuple[List[Any], Dict[str, Any]]:
    """Parse and validate inputs (prompts and functions) from files."""
    try:
        prompts = _parse_prompts(input_path)
        functions = _parse_functions(functions_path)
        return prompts, functions
    except (FileNotFoundError, PermissionError, OSError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def run_generation_loop(
    llm: Small_LLM_Model,
    vocab: Vocab,
    prompts: List[Any],
    functions_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run the constrained JSON decoding loop over all input prompts."""
    results: List[Dict[str, Any]] = []
    print(f"[*] Processing {len(prompts)} prompts sequentially...")

    for item in prompts:
        print(f" -> Generating for prompt: '{item.prompt}'")
        json_output_string = generate_json(
            llm, vocab, item.prompt, functions_dict
        )
        try:
            parsed_object: Dict[str, Any] = json.loads(json_output_string)
            results.append(parsed_object)
        except json.JSONDecodeError as e:
            print(
                f"Error: Constructed invalid JSON string: {e}",
                file=sys.stderr,
            )
            continue

    return results


def save_results(output_path: str, results: List[Dict[str, Any]]) -> None:
    """Save generation results to target file."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Mandatory pipeline completed. Output saved to {output_path}")


def main() -> None:
    """Main execution orchestrator."""
    start_time = time.time()
    args = parse_arguments()
    prompts, functions_dict = parse_inputs(
        args.input, args.functions_definition
    )

    # Initialize the local LLM SDK wrappers
    llm = Small_LLM_Model()
    vocab = Vocab(llm)

    results = run_generation_loop(llm, vocab, prompts, functions_dict)
    save_results(args.output, results)

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"[*] Total execution time: {elapsed_min:.2f} minutes")


if __name__ == "__main__":
    main()
