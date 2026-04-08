"""CLI entry point: parse inputs, load the model, run constrained
decoding for every prompt, and write the results to JSON."""

import argparse
import json
import os
import sys
import time
from typing import List

from llm_sdk import Small_LLM_Model
from src.decoder import Decoder, JsonObject
from src.parsing import parse_functions, parse_prompts
from src.vocab import Vocab

MODEL_CHOICES = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"]
DEFAULT_MODEL = MODEL_CHOICES[0]
DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def write_results(path: str, results: List[JsonObject]) -> None:
    """Write the results as pretty-printed JSON.

    Parent directories are created if needed.

    Args:
        path: Destination file path.
        results: The generated function-call objects, in order.

    Returns:
        None.

    Raises:
        OSError: If the directory or file cannot be written.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments.

    Returns:
        The parsed namespace: functions_definition, input, output,
        verbose, and model.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe — function calling"
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each generation step to stderr (bonus).",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
        help=f"Model to use (bonus, default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline: parse -> load model -> decode -> write.

    Returns:
        None. Exits the process with status 1 on a fatal error;
        failed prompts are skipped, then status 1 reports partial failure.
    """
    args = parse_args()
    try:
        prompts = parse_prompts(args.input)
        functions = parse_functions(args.functions_definition)
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    if not prompts:
        try:
            write_results(args.output, [])
        except OSError as error:
            print(f"Error: cannot write {args.output}: {error}",
                  file=sys.stderr)
            sys.exit(1)
        return

    start = time.time()
    try:
        llm = Small_LLM_Model(model_name=args.model)
        vocab = Vocab(llm)
        decoder = Decoder(
            functions=functions, vocab=vocab, verbose=args.verbose
        )
    except Exception as error:
        print(
            f"Error: failed to initialize model {args.model!r}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)

    results: List[JsonObject] = []
    print(f"[*] Model: {args.model}")
    print(f"[*] Processing {len(prompts)} prompts...")
    for i, item in enumerate(prompts, 1):
        try:
            print(f"  [{i}/{len(prompts)}] {item.prompt!r}")
            results.append(decoder.run(item.prompt))
        except Exception as error:
            print(f"Error on {item.prompt!r}: {error}", file=sys.stderr)

    try:
        write_results(args.output, results)
    except OSError as error:
        print(f"Error: cannot write {args.output}: {error}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    print(f"[*] Done: {len(results)}/{len(prompts)} in {elapsed:.1f}s")
    if len(results) != len(prompts):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt ;)", file=sys.stderr)
        sys.exit(1)
    except ImportError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"Error: unexpected failure: {error}", file=sys.stderr)
        sys.exit(1)
