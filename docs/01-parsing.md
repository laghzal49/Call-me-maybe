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

`_load_json_array(path)` is the shared loader:

1. Open with a context manager (auto-closes the file).
2. Catch `FileNotFoundError`, `OSError`, `JSONDecodeError` and re-raise as a
   `ValueError` with a readable message.
3. Check the top level is a `list`; otherwise raise.

`parse_prompts` / `parse_functions` then validate each item:

- Each entry must be a dict, else "entry N is not an object".
- Build the pydantic model; a `ValidationError` becomes a `ValueError` naming the
  bad entry.
- `parse_functions` additionally rejects **duplicate names** and requires **at
  least one** function.

## Why this design

- **Why pydantic?** The subject requires it, and it gives validation + types for
  free instead of hand-written `if` checks.
- **Why a dict for functions (not a list)?** Generation looks up the chosen
  function by name many times; a dict makes that O(1) and rejects duplicates.
- **Why convert every error to `ValueError`?** `__main__` then catches a single
  exception type and prints one clean message — no tracebacks for the reviewer.

## How to reimplement

1. Define the three pydantic models exactly as above.
2. Write one helper that reads a file inside `try/except`, parses JSON, and
   asserts the root is a list.
3. Write two functions that loop over the list, validate each item, and collect
   results (a list for prompts, a dict keyed by name for functions).
4. Add the duplicate-name and non-empty checks for functions.

## Edge cases

- Missing file, unreadable file, invalid JSON, non-array root → clear `ValueError`.
- A prompt entry missing the `prompt` key, or a function missing `parameters` →
  pydantic `ValidationError` → `ValueError` naming the entry index.
