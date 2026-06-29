"""Command-line entry point and orchestration.

uv run python -m src [--functions_definition <file>] [--input <file>]
                     [--output <file>]
"""

import argparse
import sys
import time
from typing import List

from llm_sdk import Small_LLM_Model

from src.context import build_generation_context
from src.output import validate_result, write_results
from src.parsing import parse_functions, parse_prompts
from src.state_machine import JsonObject, StateMachine
from src.vocab import Vocab

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Define and parse the three optional path arguments."""
    parser = argparse.ArgumentParser(description="Call Me Maybe — function calling")
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the pipeline: parse -> load model -> decode -> validate -> write."""
    args = parse_args()
    start = time.time()
    try:
        prompts = parse_prompts(args.input)
        functions = parse_functions(args.functions_definition)
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    try:
        llm = Small_LLM_Model()
        vocab = Vocab(llm)
        ctx = build_generation_context(llm, functions)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: failed to initialize the model: {error}", file=sys.stderr)
        sys.exit(1)
    results: List[JsonObject] = []
    print(f"[*] Processing {len(prompts)} prompts...")
    for i, item in enumerate(prompts, 1):
        try:
            print(f"  [{i}/{len(prompts)}] {item.prompt!r}")
            result = StateMachine(llm, vocab, ctx, item.prompt).run()
            validate_result(result, functions)
            results.append(result)
        except (ValueError, KeyError) as error:
            print(f"Error on {item.prompt!r}: {error}", file=sys.stderr)

    try:
        write_results(args.output, results)
    except OSError as error:
        print(f"Error: cannot write {args.output}: {error}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    print(f"[*] Done: {len(results)}/{len(prompts)} calls in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
