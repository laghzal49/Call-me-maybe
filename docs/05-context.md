# 05 — `src/context.py`

## Role

Build, **once per run**, everything that is the same for every prompt: the text
block describing the available functions, and the tries used to constrain the
function name and boolean values.

## Theory

Some work does not depend on the specific prompt — encoding the function names
into a trie, rendering the function list as text. Doing it once and reusing it for
all prompts saves time and keeps the per-prompt code small. This is a classic
"precompute the invariant part" optimization.

## Inputs / Outputs

- **Input:** the SDK model and the `Dict[str, FunctionDefinition]` from parsing.
- **Output:** a `GenerationContext` holding:
  - `functions` — the same dict (for name → definition lookup).
  - `functions_block` — the prompt text listing each function.
  - `function_trie` — a `Trie` of all function names.
  - `boolean_trie` — a `Trie` of `true` / `false`.
- Also exports `INSTRUCTION`, the prompt header template.

## How it works

- `INSTRUCTION` is a template string with `{block}` and `{prompt}` placeholders:

  ```
  You convert the request into a function call.
  Available functions:
  {block}
  Request: {prompt}
  ```

- `build_functions_block(functions)` renders one line per function, e.g.
  `- fn_add_numbers(a: number, b: number): Add two numbers...`. This is what tells
  the model what it can call and what each argument means.
- `GenerationContext` is a **pydantic** model. Because a `Trie` is not a pydantic
  type, it sets `model_config = ConfigDict(arbitrary_types_allowed=True)`.
- `build_generation_context(llm, functions)` fills all fields, building the two
  tries with `Trie.from_strings`.

## Why this design

- **Why keep `functions` in the context?** After the model picks a name, we need
  its parameter schema; the dict gives it in O(1).
- **Why include descriptions in the block?** The model chooses the function from
  this text, so a short description improves accuracy. Names alone are often
  enough here, but descriptions add a safety margin.
- **Why pydantic for a bundle of objects?** Consistency with the subject's "all
  classes use pydantic" rule and a single typed container to pass around.

## How to reimplement

1. Define the `INSTRUCTION` template.
2. Write `build_functions_block` to join one signature line per function.
3. Define `GenerationContext` (allow arbitrary types) with the four fields.
4. Write `build_generation_context` that renders the block and builds both tries.

## Edge cases

- Empty function list is impossible here — `parse_functions` already requires at
  least one function.
