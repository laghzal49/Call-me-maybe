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
    """Write the results list as pretty-printed JSON.

    Args:
        path: Destination file path; parent directories are created
            if they do not exist.
        results: List of function-call result objects to serialize.

    Returns:
        None.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments.

    Args:
        None (reads from ``sys.argv`` via argparse).

    Returns:
        The parsed arguments: ``functions_definition``, ``input``,
        ``output``, ``verbose``, and ``model``.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe — function calling"
    )
    parser.add_argument(
        "--functions_definition", default=DEFAULT_FUNCTIONS
    )
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

    Reads prompts and function definitions from the paths given on
    the CLI (or their defaults), loads the model, runs constrained
    decoding for every prompt, and writes the collected results to
    the output JSON file.

    Args:
        None (reads configuration from ``parse_args()``).

    Returns:
        None. Exits the process with status 1 on a fatal error.
    """
    args = parse_args()
    try:
        prompts = parse_prompts(args.input)
        functions = parse_functions(args.functions_definition)
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    try:
        llm = Small_LLM_Model(model_name=args.model)
        start = time.time()
        vocab = Vocab(llm)
        decoder = Decoder(
            functions, vocab=vocab, verbose=args.verbose
        )
    except Exception as error:
        print(
            f"Error: failed to initialize model "
            f"{args.model!r}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)

    results: List[JsonObject] = []
    print(f"[*] Model: {args.model}")
    print(f"[*] Processing {len(prompts)} prompts...")
    for i, item in enumerate(prompts, 1):
        try:
            print(f"  [{i}/{len(prompts)}] {item.prompt!r}")
            result = decoder.run(item.prompt)
            results.append(result)
        except Exception as error:
            print(
                f"Error on {item.prompt!r}: {error}", file=sys.stderr
            )

    try:
        write_results(args.output, results)
    except OSError as error:
        print(
            f"Error: cannot write {args.output}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)

    elapsed = time.time() - start
    print(
        f"[*] Done: {len(results)}/{len(prompts)} in {elapsed:.1f}s"
    )


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
