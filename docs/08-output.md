# 08 — `src/output.py`

## Role

The final safety net: check each generated call against the function schema, then
write all results to the output JSON file.

## Theory

Constrained decoding already guarantees structural validity, but a separate
**validation** step enforces the contract independently ("trust, but verify"). If
a result ever breaks the schema, we want to know — not ship it silently. Writing
is done with a context manager so the file is always closed, and `json.dump`
guarantees valid, well-escaped JSON.

## Inputs / Outputs

| | Input | Output |
|---|-------|--------|
| `validate_result(result, functions)` | one call dict + the function defs | nothing, or raises `ValueError` |
| `write_results(path, results)` | output path + list of call dicts | writes the JSON file |

## How it works

`validate_result` checks, in order (matching the subject's V.4.2 rules):

1. Keys are **exactly** `{prompt, name, parameters}` — no extra keys.
2. `name` is a **known** function.
3. `parameters` has **exactly** the function's declared parameter names.
4. Each value's Python type matches its declared type:
   - `string → str`, `boolean → bool`,
   - `number`/`integer → int or float`, but **not** `bool` (in Python `bool` is a
     subclass of `int`, so it is rejected explicitly to avoid `True` passing as a
     number).

Any failure raises a `ValueError` with a precise message (e.g.
`"fn_add_numbers.a: expected number"`).

`write_results`:

1. Create the parent directory with `os.makedirs(..., exist_ok=True)` if needed.
2. Open the file with a `with` block (auto-close) and `json.dump(results, ...,
   indent=2)`, then a trailing newline.

## Why this design

- **Why validate after constrained decoding?** Defense in depth. The decoder
  *should* be correct; validation proves it and catches any future regression.
- **Why reject `bool` for numbers explicitly?** Because `isinstance(True, int)` is
  `True` in Python; without the explicit check a boolean could sneak into a number
  field.
- **Why `json.dump` (not manual string building)?** It guarantees valid JSON and
  correct escaping (quotes, backslashes, unicode) for free.

## How to reimplement

1. Write `validate_result` with the four checks above; raise `ValueError` with a
   clear message on the first failure.
2. Write `write_results` to ensure the directory exists and dump the list as
   pretty JSON inside a context manager.

## Edge cases

- Output directory does not exist → created automatically.
- A bad result → `validate_result` raises; `__main__` catches it per-prompt, prints
  a message, and keeps going (one bad prompt does not fail the batch).
- Write failure (permissions, disk) → `OSError`, caught in `__main__`.
