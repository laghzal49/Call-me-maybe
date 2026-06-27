import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model
from src.decode import build_generation_context, generate_call
from src.parsing import FunctionDefinition, Prompt, parse_functions
from src.parsing import parse_prompts
from src.vocab import Vocab


def main() -> None:
    """Main entry point: parse inputs, run constrained decoding, save."""
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
    args = parser.parse_args()

    start = time.time()

    try:
        prompts: List[Prompt] = parse_prompts(args.input)
        functions: Dict[str, FunctionDefinition] = parse_functions(
            args.functions_definition
        )
    except (FileNotFoundError, PermissionError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    try:
        llm = Small_LLM_Model()
        vocab = Vocab(llm=llm)
        context = build_generation_context(llm, functions)
    except (OSError, ValueError, RuntimeError) as error:
        print(
            f"Error: failed to initialize the model: {error}",
            file=sys.stderr,
        )
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    print(f"[*] Processing {len(prompts)} prompts sequentially...")
    for item in prompts:
        print(f" -> Generating for prompt: '{item.prompt}'")
        try:
            results.append(generate_call(llm, vocab, item.prompt, context))
        except ValueError as error:
            print(
                f"Error: failed to process prompt {item.prompt!r}: {error}",
                file=sys.stderr,
            )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    elapsed = (time.time() - start) / 60.0
    print(f"[+] Mandatory pipeline completed. Output saved to {args.output}")
    print(f"[*] Total execution time: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
