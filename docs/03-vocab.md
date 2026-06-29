# 03 — `src/vocab.py`

## Role

Load the model's vocabulary and precompute the **token-id sets** needed to mask
logits while decoding numbers and strings.

## Theory

To constrain a number, we must know *which token ids are digits*. To stop a
string, we must know *which token id is the closing quote*. The model only gives
us logits indexed by token id, so we need a map from **token id → its text**, and
from that we build small reusable sets. We do this **once** at startup; scanning
the ~150k-token vocabulary on every generated token would be far too slow.

The subject hints at exactly this: *"use the vocabulary JSON file to map between
tokens and their string representations ... to determine which tokens are valid."*

## Inputs / Outputs

- **Input:** the SDK model (used for `get_path_to_vocab_file`, `encode`, `decode`).
- **Output (attributes built in `__init__`):**
  - `digit_ids: Set[int]` — tokens made only of digits.
  - `dot_id`, `minus_id: int` — the `.` and `-` tokens (`-1` if absent).
  - `number_end_ids: Set[int]` — tokens that legally end a number (`, } ] ` space, newline).
  - `quote_id: int` — the closing `"` token.
  - `string_forbidden_ids: Set[int]` — tokens that contain a `"` (except the clean
    closing quote), which must never appear inside a string value.
- **Methods:** `decode_token(id)`, `number_tokens(...)`, `integer_tokens(...)`.

## How it works

1. `_load_json()` reads the path from `llm.get_path_to_vocab_file()` (a JSON map
   `text -> id`) inside a context manager, with clear errors on missing/invalid
   files. We also build the reverse `id_to_text`.
2. Classify in one pass:
   - `digit_ids` = ids whose text is all of `0-9` (single- **and** multi-digit
     tokens like `"42"` both keep a number valid and make generation faster).
   - `dot_id` / `minus_id` via direct lookup (`.` and `-` are plain ASCII, so they
     appear verbatim as vocab keys).
   - `quote_id` = lookup of `"`; `string_forbidden_ids` = every id whose text
     contains `"`, minus the clean `quote_id`.
3. `_encode_first([... ])` encodes each end character and keeps its first token id
   → `number_end_ids`. We use `encode` here (not vocab keys) because characters
   like space are stored byte-encoded in the raw vocab, and `encode` handles that.
4. `number_tokens(started, has_digit, has_dot)` returns the valid ids at a numeric
   step: digits always; a sign **only when nothing emitted yet** (`not started`);
   a dot **only once and only after a digit**. `integer_tokens(started)` is the
   same but never allows a dot.

## Why this design

- **Why precompute sets?** Speed. Classify once, reuse on every step.
- **Why `started` instead of `has_digit` for the sign?** A bug-fix: with
  `has_digit` the sign stayed legal after a `-`, allowing `--5`. `started` allows
  the sign only as the very first character, so `--5`, `-.`, etc. are impossible.
- **Why exclude only quote tokens from strings (not backslashes/controls)?** We
  build a Python string and the output stage runs `json.dump`, which escapes
  everything. Allowing backslashes is essential — regex parameters like `\d+` must
  be expressible, and `json.dump` turns them into valid `\\d+`.
- **Why `decode_token` via `llm.decode`?** The raw vocab keys are byte-encoded
  (e.g. the `Ġ` leading-space marker); `llm.decode([id])` gives clean text, which
  is what we accumulate into a value.

## How to reimplement

1. Load the vocab JSON to a `text -> id` dict; build the reverse map.
2. One pass: collect digit-only ids; look up `.`, `-`, `"`; collect quote-bearing
   ids as forbidden (minus the clean quote).
3. Encode the number-end characters to get their token ids.
4. Implement `number_tokens` / `integer_tokens` with the started/digit/dot rules.

## Edge cases

- Vocab without a `.` or `-` token → `dot_id`/`minus_id` are `-1` and simply never
  added to an allowed set.
- A multi-digit token (`"265"`) is in `digit_ids`, so a big number can be one
  token → fewer model calls.
