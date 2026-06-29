# 00 — Overview & Reading Order

This folder explains the project **one file at a time**. Each document covers the
same six things for its file:

1. **Role** — one sentence.
2. **Theory** — the idea behind it.
3. **Inputs / Outputs** — what goes in, what comes out.
4. **How it works** — a walkthrough of the code.
5. **Why this design** — rationale and rejected alternatives.
6. **How to reimplement** — steps to rebuild it from scratch.

## The problem we are solving

A small language model (Qwen3-0.6B) is given a natural-language request such as
*"What is the sum of 2 and 3?"*. We must output a **function call** as JSON:

```json
{"prompt": "What is the sum of 2 and 3?",
 "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

Small models are unreliable at producing valid JSON if you just ask them. So we
use **constrained decoding**: we generate token by token and, at every step,
forbid any token that would break JSON structure or the function's type schema.

## The one big idea: forced vs. free positions

While building the JSON, every position is one of two kinds:

- **Forced** — only one token is ever valid (the `{`, the `"`, the `:` between a
  key and its value). We just *write* these ourselves; running the model would
  only confirm the single legal choice.
- **Free** — the model genuinely chooses: *which* function, *which* digits,
  *which* string text. Here we run the model, mask the logits down to the valid
  tokens, and let it pick.

So: **structure is forced, content is masked.** This guarantees 100% valid JSON
and lets the model make only the choices that matter.

---

## Full pipeline diagram

```
INPUTS
══════
functions_definition.json ──▶ parse_functions() ──▶ Dict[name, FunctionDefinition]
                                                              │
function_calling_tests.json ──▶ parse_prompts() ──▶ List[Prompt]
                                                              │
                                                              ▼
                              STARTUP (once per run)
                              ═════════════════════
                          Small_LLM_Model()  ──▶  llm
                          Vocab(llm)         ──▶  vocab   (token-id sets)
                          build_generation_context(llm, functions)
                                ├── functions_block  (text injected into every prompt)
                                ├── function_trie    (token paths for every function name)
                                └── boolean_trie     (token paths for "true" / "false")

                                              │
                                              ▼
                              PER PROMPT  (StateMachine)
                              ═════════════════════════════
          ┌───────────────────────────────────────────────────────────────┐
          │                                                               │
          │  self.ids = encode(INSTRUCTION + '{"name": "')               │
          │                                                               │
          │   ┌──────────────────────────────────────────────────────┐   │
          │   │  STATE: NAME                                         │   │
          │   │  walk_trie(function_trie)                            │   │
          │   │    ├─ forced steps: only 1 child  → append, no model │   │
          │   │    └─ branching step: model picks from valid children │   │
          │   │  emit('", "parameters": {')                          │   │
          │   └──────────────────────────────────────────────────────┘   │
          │                         │                                     │
          │                         ▼                                     │
          │   ┌──────────────────────────────────────────────────────┐   │
          │   │  STATE: PARAMS                                       │   │
          │   │  for each parameter (in schema order):              │   │
          │   │                                                      │   │
          │   │    emit('"key": ')                                   │   │
          │   │                                                      │   │
          │   │    string  → decode_string()                         │   │
          │   │               loop: pick_excluding(quote_ids)        │   │
          │   │               stop: quote token wins over content    │   │
          │   │                                                      │   │
          │   │    boolean → walk_trie(boolean_trie)                 │   │
          │   │               constrained to "true" or "false"       │   │
          │   │                                                      │   │
          │   │    integer → decode_number(integer_only=True)        │   │
          │   │    number  → decode_number(integer_only=False)       │   │
          │   │               allowed: digits [+sign at start]       │   │
          │   │               [+dot once, for number]                │   │
          │   │               stop: end-token (,  }  ]  …) wins     │   │
          │   │                                                      │   │
          │   │    emit(', ')  between params; emit('}}') at end     │   │
          │   └──────────────────────────────────────────────────────┘   │
          │                         │                                     │
          │                         ▼                                     │
          │               STATE: DONE  →  return JsonObject              │
          └───────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                              OUTPUT
                              ══════
                          validate_result()   (schema safety check)
                          write_results()     (json.dump → guaranteed valid JSON)
                                              │
                                              ▼
                    data/output/function_calling_results.json


CONSTRAINED DECODING — how a single step works
═══════════════════════════════════════════════

  LLM produces logits[0 .. vocab_size-1]  (one float per possible next token)
         │
         ▼  MASK  (set forbidden tokens to −∞)
         │
  Only the valid tokens survive
         │
         ▼  argmax  (greedy pick)
         │
  best_token  →  appended to self.ids  →  becomes input for the next step

  What counts as "valid" depends on the current generation context:

  ┌────────────────┬────────────────────────────────────────────────────┐
  │ Context        │ Valid tokens                                       │
  ├────────────────┼────────────────────────────────────────────────────┤
  │ function name  │ children of current trie node                     │
  │ boolean value  │ children of current trie node ("true" / "false")  │
  │ string content │ all tokens whose text does NOT start with "        │
  │ string close   │ first token among those that DO start with "       │
  │ integer digit  │ digit-only tokens; minus only at position 0        │
  │ number digit   │ digit tokens; minus at pos 0; one dot after digit  │
  │ number/int end │ end-tokens (, } ] space …) after at least 1 digit  │
  └────────────────┴────────────────────────────────────────────────────┘
```

---

## Pipeline (reading order)

```
input files ─▶ parsing ─▶ build context (tries + text) ─▶ for each prompt:
                                                              StateMachine.run()
                                                              ├─ pick name (trie)
                                                              └─ decode each value
                                                          ─▶ validate ─▶ write JSON
```

Read the files in this order:

| Doc | File | What it adds |
|-----|------|--------------|
| [01](01-parsing.md) | `src/parsing.py` | Load & validate the input JSON files |
| [02](02-trie.md) | `src/trie.py` | Trie data structure for fixed-choice tokens |
| [03](03-vocab.md) | `src/vocab.py` | Which token ids are digits / quotes / etc. |
| [04](04-masking.md) | `src/masking.py` | Pick one token from the valid set |
| [05](05-context.md) | `src/context.py` | Precompute tries + prompt text once |
| [06](06-state_machine.md) | `src/state_machine.py` | The token-by-token driver |
| [07](08-output.md) | `src/output.py` | Validate against schema + write file |
| [08](09-main.md) | `src/__main__.py` | CLI and orchestration |

`src/__init__.py` only marks `src` as a package so `python -m src` works.

## Rules we respect (from the subject)

- Constrained decoding via **logit masking**, never "prompt and hope".
- The function is chosen **by the model**, not by keyword heuristics.
- Only **public** `llm_sdk` methods are used.
- All data classes use **pydantic**; code passes **flake8** and **mypy**.
- Output is **always valid JSON** matching the schema.
