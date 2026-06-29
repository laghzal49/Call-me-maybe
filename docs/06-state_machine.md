# 06 — `src/state_machine.py`

## Role

The engine. It builds **one** function call by extending a single token sequence,
running the model only where there is a real choice and masking the logits so
only valid tokens can be picked.

## Theory

### One growing sequence ("track and continue")

The model is autoregressive: to predict the next token it needs the whole
sequence so far. We keep one list, `self.ids`, and **only append** to it — the
function name we choose, the value tokens the model picks, and the structural
tokens we write. We never rebuild or re-encode the prompt mid-generation, so the
sequence faithfully *continues* rather than being reconstructed.

### Structure is written, content is masked

We are in control of the JSON shape, so we *write* the fixed parts (`{`, `"`,
keys, `:`, `,`, `}`) directly with `emit()`. We only call the model for:

- the **function name** (constrained by the function trie), and
- each **value** (constrained by its declared type).

### A state machine

Generation has clear phases, so we model it as states:

```
NAME ──▶ PARAMS ──▶ DONE
```

`run()` advances through them. Tiny, but it makes the flow explicit and easy to
extend.

## Inputs / Outputs

- **Input:** `llm`, `vocab`, `ctx` (the `GenerationContext`), and the `prompt`.
- **Output:** `run()` returns `{"prompt", "name", "parameters"}` — a Python dict,
  which the output stage turns into guaranteed-valid JSON.

## How it works

### Setup (`__init__`)

Builds the starting sequence: the instruction header (with the functions block
and the request) followed by the literal `{"name": "`. So the model's very next
prediction is the first token of the function name. Also initializes
`result = {"prompt": prompt, "name": "", "parameters": {}}` and `state = NAME`.

### Helpers

- `emit(text)` — encode fixed text and append it. No model call: these are forced
  tokens.
- `logits()` — `llm.get_logits_from_input_ids(self.ids)`: the next-token logits for
  the current sequence.

### `walk_trie(root)` — name & boolean

Descends the trie. **Key optimization:** if the current node has exactly **one**
child, that token is *forced*, so we append it without calling the model. Only at
a real branch (2+ children) do we call `logits()` and `pick_allowed(...)` over the
children. Returns the word stored on the leaf. (All function names share the
`fn_` prefix — that whole prefix is forced, so name selection costs a model call
only at the point where names actually diverge.)

### `decode_string()`

Loops up to `MAX_STRING_TOKENS`. Each step uses `pick_excluding(logits,
string_forbidden_ids)`: the best token that does **not** contain a quote. If the
model picks the closing-quote token, the string is finished. Otherwise we decode
the token to text, append it, and continue. The opening and closing quotes
themselves are written by `do_params` (structure), not generated here.

### `decode_number(integer_only)`

Tracks `started`, `has_digit`, `has_dot`. Each step asks `vocab` for the allowed
numeric tokens; once at least one digit exists, the **number-end tokens** are
added to the allowed set so the model can choose to stop. If the pick is an end
token, the number is done (we do not append it — the comma/brace is structure). A
multi-digit token counts as digits; a `.` flips `has_dot`. Finally parse to `int`
(integer) or `float` (number); if no digit was produced, default to `0`.

### `decode_value(schema)`

Dispatch by type: `string → decode_string`, `boolean → walk_trie(boolean_trie)`
compared to `"true"`, otherwise `decode_number` with `integer_only` set for the
`integer` type.

### Phases

- `do_name()` — walk the function trie, store the name (fallback to the first
  function if empty), then `emit('", "parameters": {')`.
- `do_params()` — for each parameter in order: `emit('"key": ')`, open a quote for
  strings, decode the value, close the quote for strings, record it in `result`,
  and `emit(", ")` before the next one. After the last, `emit("}}")` to close the
  parameters object and the root.
- `run()` — `NAME → do_name`, `PARAMS → do_params`, then `DONE`; return `result`.

## Why this design

- **Why write structure instead of masking it?** At a structural position only one
  token is valid; masking would force it anyway. Writing it is identical and skips
  a model call — the same "fast-forward forced tokens" trick used in `walk_trie`.
- **Why build a dict and not emit raw JSON text?** A real dict + `json.dump`
  cannot produce malformed JSON. Structure validity is free.
- **Why stop a number with an "end token" choice (not by peeking at the
  unconstrained best)?** So termination is itself a constrained decision: the
  model picks from `digits ∪ end-tokens`, all valid. We never read an unmasked
  prediction.
- **Why caps (`MAX_*`)?** Safety: a degenerate model can never loop forever.

## Speed notes

The SDK recomputes the whole prefix on every `get_logits` call (no cache), so cost
≈ number of model calls. We minimize calls by:

- never calling the model for structure (only `emit`),
- skipping forced (single-child) trie steps in `walk_trie`,
- allowing multi-digit tokens so numbers take fewer steps,
- deciding booleans in effectively one branching call.

## How to reimplement

1. In `__init__`, encode `header + '{"name": "'` into `self.ids`; set up `result`.
2. Write `emit` (append encoded text) and `logits` (call the SDK).
3. Write `walk_trie` with the single-child fast-forward.
4. Write `decode_string` (mask out quote tokens, stop on the close quote).
5. Write `decode_number` (numeric mask + end tokens, parse at the end).
6. Write `decode_value` to dispatch by `schema.type`.
7. Write `do_name`, `do_params`, and a `run` loop over the `State` enum.

## Edge cases

- Model emits no digits for a number → value defaults to `0` (then schema
  validation still passes for number/integer).
- A function with no parameters → the loop is skipped and `emit("}}")` yields
  `"parameters": {}`.
- A string value containing a backslash (e.g. a regex) is allowed during
  generation and escaped correctly by `json.dump` at output time.
