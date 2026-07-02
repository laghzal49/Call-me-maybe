"""
CONCEPT 10 — FULL PIPELINE (needs model)
=========================================
This test runs the complete Decoder end-to-end and shows how all the
previous concepts combine into a single coherent system.

PIPELINE DIAGRAM:
─────────────────
                       Prompt text
                           │
                     encode(prompt)
                           │  token ids
                    ┌──────▼──────┐
                    │  fn_trie    │  CONCEPT 4+9: trie walk over fn names
                    │   walk      │  CONCEPT 3: _pick() at branch points
                    └──────┬──────┘
                     function name
                           │
               ┌───────────▼────────────┐
               │  per-parameter decode  │
               │  ├── string:  _string() │  CONCEPT 6: quote race
               │  ├── boolean: trie walk │  CONCEPT 8
               │  └── number:  _number() │  CONCEPT 7: digit accumulation
               └───────────┬────────────┘
                           │
                      validate_result      (check schema correctness)
                           │
                       {"name": ..., "parameters": {...}}

WHAT THE MODEL DOES:
  - Provides logits at every branching point (trie) or every token (strings/numbers)
  - Its logits encode which function name fits the prompt
  - Its logits encode what value fits each parameter

WHAT THE CODE DOES:
  - Constrains which tokens the model may pick (never none, always valid JSON)
  - Guarantees the output matches the schema, even if the model "wants" to deviate

Run: uv run python tests/test_10_full_pipeline.py
     (requires the model to be downloaded)
"""

import sys

sys.path.insert(0, ".")

from llm_sdk import Small_LLM_Model  # noqa: E402
from src.parsing import parse_functions  # noqa: E402
from src.decoder import Decoder  # noqa: E402
from src.output import validate_result  # noqa: E402

# ── 1. setup (done ONCE, not per prompt) ─────────────────────────────────────
# This mirrors __main__.py.
# Decoder.__init__ loads vocab, builds tries and token sets.

llm = Small_LLM_Model()
functions = parse_functions("data/input/functions_definition.json")
decoder = Decoder(llm, functions)

print("Functions available:")
for name, fn in functions.items():
    params = {k: v.type for k, v in fn.parameters.items()}
    print(f"  {name}: {params}")

# ── 2. run a few prompts through the pipeline ─────────────────────────────────
# Each call to decoder.run() generates one JSON object.
# The model picks the function; the trie + constraints produce valid JSON.

test_cases = [
    "What is the sum of 10 and 5?",
    "Greet Alice",
    "Reverse the string hello",
    "What is the square root of 9?",
    "Is the number 4 positive?",
]

print()
for prompt in test_cases:
    result = decoder.run(prompt)
    try:
        validate_result(result, functions)
        error = None
    except ValueError as exc:
        error = exc

    status = "OK" if error is None else "FAIL"
    print(f"[{status}] prompt  : {prompt!r}")
    print(f"       name    : {result['name']!r}")
    print(f"       params  : {result['parameters']}")
    if error is not None:
        print(f"       ERROR   : {error}")
    print()

# ── 3. what to implement to recode this project from scratch ──────────────────
print("=" * 70)
print("RECAP: what you need to recode this project")
print("=" * 70)
print("""
1. VOCAB (test_05)
   vocab = json.load(open(llm.get_path_to_vocab_file()))
   digit_ids   = {i for t,i in vocab.items() if t and all(c.isdigit() for c in t)}
   dot_id      = vocab.get(".", -1)
   minus_id    = vocab.get("-", -1)
   quote_ids   = {i for t,i in vocab.items() if '"' in t}
   end_ids     = {encode_ids(llm, c)[0] for c in (",","}"," ")}

2. TRIE (test_04)
   insert(token_id_list, word_value) into a tree of TrieNode
   from_strings(llm, words) = encode each word, then insert

3. _pick(lg, allowed) (test_03)
   masked = np.full(len(lg), -inf)
   masked[list(allowed)] = lg[list(allowed)]
   return int(np.argmax(masked))

4. _walk(trie_root) (test_04, test_09)
   while node.children:
       if 1 child: forced — take it
       else:       _pick(logits(), node.children.keys())
   return node.value

5. _string() (test_06)
   loop:
       lg = logits()
       close = _pick(lg, quote_ids)
       close_val = float(lg[close])   # BEFORE masking
       lg[list(quote_ids)] = -inf
       best = argmax(lg)
       if close_val >= lg[best]: salvage prefix before "; break
       else: append best token

6. _number(integer_only) (test_07)
   loop:
       allowed = digit_ids
       if not text: add minus_id
       if has_digit and not has_dot and not integer_only: add dot_id
       if has_digit: add end_ids
       tok = _pick(logits(), allowed)
       if tok in end_ids: break
       append tok

7. run(prompt) — puts it all together
   ids = encode(header + '{"name": "')
   name = _walk(fn_trie)
   emit('"name": "' + name + '", "parameters": {')
   for key, schema in function.parameters:
       emit('"' + key + '": ')
       if string:  emit('"'); value = _string(); emit('"')
       if boolean: value = _walk(bool_trie) == "true"
       if number:  value = _number(integer_only=schema.type=="integer")
       if not last: emit(", ")
   emit("}}")
   return {"prompt": prompt, "name": name, "parameters": params}
""")
