# File Guide

This document explains what each project file does, what it receives, what it
returns or creates, and why it exists.

## Root Files

### `README.md`

- What it does: explains the project, how to install it, how to run it, the
  constrained decoding algorithm, design choices, performance notes, testing,
  and resources.
- Input: none at runtime.
- Output: documentation for reviewers and peers.
- Why this file exists: the subject explicitly requires a README in English with
  these sections.

### `Makefile`

- What it does: provides the mandatory commands `install`, `run`, `debug`,
  `clean`, and `lint`.
- Input: optional `ARGS`, for example `make run ARGS="--input file.json"`.
- Output: runs setup, the program, debugger, cleanup, or lint/type checks.
- Why this design: reviewers can use the same stable commands instead of
  remembering long `uv` commands.

### `pyproject.toml`

- What it does: declares the Python project, dependencies, workspace SDK source,
  and mypy configuration.
- Input: read by `uv`, `mypy`, and Python tooling.
- Output: controls dependency installation and type-check behavior.
- Why this design: `uv sync` is required by the subject, so project metadata must
  be centralized here.

### `uv.lock`

- What it does: locks exact dependency versions resolved by `uv`.
- Input: generated from `pyproject.toml`.
- Output: reproducible dependency installation.
- Why this design: reviewers get the same dependency graph instead of a moving
  target.

### `.flake8`

- What it does: configures flake8 exclusions.
- Input: read by `flake8`.
- Output: prevents linting generated caches, virtual environments, and the
  provided SDK.
- Why this design: the project code is linted without failing on external or
  generated files.

### `.gitignore`

- What it does: ignores `.venv`, caches, logs, and generated `data/output/`.
- Input: read by git.
- Output: keeps generated files out of submission.
- Why this design: the subject says output should be generated during review and
  not committed.

### `pyrightconfig.json`

- What it does: editor/type-checker configuration for Pyright.
- Input: read by editors that support Pyright.
- Output: local editor diagnostics.
- Why this design: helpful for development, but not part of the mandatory run
  path.

## Source Files

### `src/__main__.py`

- What it does: command-line entry point for `uv run python -m src`.
- Input:
  - `--functions_definition`, defaulting to
    `data/input/functions_definition.json`
  - `--input`, defaulting to `data/input/function_calling_tests.json`
  - `--output`, defaulting to
    `data/output/function_calling_results.json`
- Output: writes one JSON array to the output path.
- Logic:
  1. Parse command-line arguments.
  2. Load and validate input files.
  3. Initialize the SDK model and vocabulary.
  4. Build reusable constrained-decoding context once.
  5. Generate one function call per prompt.
  6. Write the final JSON file.
- Why this design: the entry point stays thin. It handles program flow and error
  messages, while decoding details stay in smaller modules.

### `src/parsing.py`

- What it does: validates the two input JSON files with pydantic models.
- Input:
  - prompt JSON array, where each item has `prompt`
  - function definition JSON array, where each item has `name`, `description`,
    `parameters`, and `returns`
- Output:
  - `List[Prompt]`
  - `Dict[str, FunctionDefinition]`
- Logic: load JSON, check that the root is an array, validate every item, reject
  duplicate function names, and return typed objects.
- Why this design: invalid input is caught before model generation starts, which
  makes failures clearer and avoids crashes later.

### `src/constraints.py`

- What it does: contains reusable token-level constrained-decoding helpers.
- Input:
  - SDK model
  - function definitions
  - logits from the model
  - allowed token ids
- Output:
  - `GenerationContext`
  - token id lists
  - selected best token id
  - selected trie value, such as a function name or boolean text
- Logic:
  - `token_ids` converts SDK tensors into plain Python lists.
  - `mask_logits` chooses the highest-logit token from only the allowed ids.
  - `build_trie` stores valid strings as token paths.
  - `pick_from_trie` generates only tokens that keep a trie path valid.
  - `build_generation_context` creates the function-name trie and boolean trie
    once for the whole input file.
- Why this design instead of keyword matching: the subject requires the function
  to be chosen by the LLM. The trie lets the LLM choose from logits while still
  making invalid function names impossible.

### `src/value_decoder.py`

- What it does: decodes parameter values after the function has been selected.
- Input:
  - selected function schema
  - prompt text
  - already extracted parameters
  - SDK model logits
  - cached vocabulary groups
- Output: one Python value matching the schema type: `str`, `float`, `int`, or
  `bool`.
- Logic:
  - Booleans use the boolean trie, so only `true` or `false` can be selected.
  - Numbers only allow tokens that keep a valid numeric prefix.
  - Integer parameters reject decimal prefixes.
  - Strings generate inside a quote context and stop at an unescaped quote.
- Why this design instead of asking for full JSON: small LLMs often break JSON.
  Decoding one typed value at a time is simpler, easier to validate, and keeps
  the final JSON structure under Python control.

### `src/decode.py`

- What it does: orchestrates one full function-call generation.
- Input:
  - SDK model
  - vocabulary helper
  - one natural-language prompt
  - `GenerationContext`
- Output:
  - `generate_call`: a Python dict with exactly `prompt`, `name`, `parameters`
  - `generate_json`: compatibility helper returning that dict as JSON text
- Logic:
  1. Build the function-choice prompt.
  2. Pick a function name through the function trie.
  3. Decode every required parameter using `value_decoder.py`.
  4. Return the final object.
- Why this design: this file is now short and readable. It shows the main
  algorithm without hiding details in one large file.

### `src/vocab.py`

- What it does: loads the model vocabulary and precomputes token groups used by
  constraints.
- Input: SDK model, through its public vocabulary path method.
- Output:
  - token-to-id and id-to-token maps
  - quote token ids
  - string token ids
  - numeric token ids
  - number stop token ids
- Logic: inspect each vocabulary token once and store reusable lists/sets.
- Why this design: checking the whole vocabulary during every generated token
  would be slow. Precomputing groups makes decoding faster and cleaner.

### `src/trinode.py`

- What it does: defines a pydantic trie used for closed-set token generation.
- Input: token id sequences.
- Output: a tree where each path can represent one valid function name or boolean
  value.
- Logic: insert token ids, return allowed next ids for the current node, and move
  to the next node after a token is selected.
- Why this design: a trie is simple, fast, and directly matches the constrained
  decoding requirement for choosing among fixed strings.

### `src/__init__.py`

- What it does: marks `src` as a Python package.
- Input: none.
- Output: allows `python -m src` and `from src...` imports.
- Why this design: required for the chosen module execution style.

## Data Files

### `data/input/functions_definition.json`

- What it does: example list of available functions.
- Input: edited by the user/reviewer.
- Output: parsed into `FunctionDefinition` objects.
- Why this design: the subject says the function list can change during review,
  so the implementation must read it dynamically and not hardcode examples.

### `data/input/function_calling_tests.json`

- What it does: example list of prompts to process.
- Input: edited by the user/reviewer.
- Output: parsed into `Prompt` objects.
- Why this design: the program can be tested with different prompt sets without
  changing source code.

### `data/output/`

- What it does: generated output directory.
- Input: produced by running the program.
- Output: usually `function_calling_results.json`.
- Why this design: the subject says output should be generated during review, so
  this folder is ignored by git.

## Provided SDK

### `llm_sdk/`

- What it does: provided wrapper around the small LLM and tokenizer.
- Input: text prompts or token ids.
- Output:
  - encoded token ids
  - decoded text
  - next-token logits
  - vocabulary file path
- Why this design: the subject gives this SDK and forbids private SDK access, so
  the project uses only public methods.
