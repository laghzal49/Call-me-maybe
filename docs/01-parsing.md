# 01 — `src/parsing.py`

## Role

Load the two input JSON files and validate them into typed Python objects before
any model work begins.

## Theory

"Fail fast." If the input is malformed, we want a clear error *now*, not a
confusing crash deep inside generation. We use **pydantic** models as a contract:
if the JSON does not match the expected shape, pydantic raises and we turn that
into a clean `ValueError`. This is the only place that touches raw input JSON.

## Inputs / Outputs

| | Input | Output |
|---|-------|--------|
| `parse_prompts(path)` | path to a JSON **array** of `{"prompt": "..."}` | `List[Prompt]` |
| `parse_functions(path)` | path to a JSON **array** of function defs | `Dict[str, FunctionDefinition]` |

A function definition looks like:

```json
{"name": "fn_add_numbers",
 "description": "Add two numbers together and return their sum.",
 "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
 "returns": {"type": "number"}}
```

## How it works

Three pydantic models define the contract:

- `TypeSchema` — `type` is one of `number | integer | string | boolean`, plus an
  `optional` flag (default `False`). Using `Literal[...]` means an unknown type
  string is rejected automatically.
- `Prompt` — just `{prompt: str}`.
- `FunctionDefinition` — `name`, `description`, `parameters: Dict[str, TypeSchema]`,
  `returns: TypeSchema`.

All the JSON work is delegated to pydantic itself — no manual `json.loads` or
per-entry type checks:

- `_read_text(path)` reads the file inside a context manager and converts
  `FileNotFoundError` / `OSError` into a readable `ValueError`.
- `parse_prompts` feeds the raw text to a `TypeAdapter(List[Prompt])`, which
  validates JSON syntax, the top-level array shape, and every entry in one call.
- `parse_functions` does the same with `TypeAdapter(List[FunctionDefinition])`,
  then a plain loop rejects an **empty list** and **duplicate names** while
  building the dict keyed by name.

Any `ValidationError` is re-raised as a `ValueError` naming the file, so the
caller gets one clean message with pydantic's per-entry details inside.

## Why this design

- **Why pydantic?** The subject requires it, and it gives validation + types for
  free instead of hand-written `if` checks.
- **Why a dict for functions (not a list)?** Generation looks up the chosen
  function by name many times; a dict makes that O(1) and rejects duplicates.
- **Why convert every error to `ValueError`?** `__main__` then catches a single
  exception type and prints one clean message — no tracebacks for the reviewer.

## How to reimplement

1. Define the three pydantic models exactly as above.
2. Write one helper that reads a file inside `try/except` and turns I/O errors
   into `ValueError`.
3. Validate each file with `TypeAdapter(List[...]).validate_json(raw)`.
4. Loop over the function list to reject empty input and duplicate names while
   building the dict keyed by name.

## Edge cases

- Missing file, unreadable file, invalid JSON, non-array root → clear `ValueError`.
- A prompt entry missing the `prompt` key, or a function missing `parameters` →
  pydantic `ValidationError` → `ValueError` naming the entry index.
