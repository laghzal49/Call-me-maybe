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
                    │  choose()   │  CONCEPT 9: pick among fn-name token paths
                    │  over names │  CONCEPT 3: pick() where paths diverge
                    └──────┬──────┘
                     function name
                           │
               ┌───────────▼──────────────┐
               │  per-parameter decode    │
               │  ├── string:  gen_string()│  CONCEPT 6: quote race
               │  ├── boolean: choose()    │  CONCEPT 8
               │  └── number:  gen_number()│  CONCEPT 7: digit accumulation
               └───────────┬──────────────┘
                           │
                      validate_result      (check schema correctness)
                           │
                       {"name": ..., "parameters": {...}}

WHAT THE MODEL DOES:
  - Provides logits where names diverge and at every token of strings/numbers
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
# Decoder.__init__ loads the vocab and groups the token ids.

llm = Small_LLM_Model()
functions = parse_functions("data/input/functions_definition.json")
decoder = Decoder(llm, functions)

print("Functions available:")
for name, fn in functions.items():
    params = {k: v.type for k, v in fn.parameters.items()}
    print(f"  {name}: {params}")

# ── 2. run a few prompts through the pipeline ─────────────────────────────────
# Each call to decoder.run() generates one JSON object.
# The model picks the function; the constraints produce valid JSON.

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
   digit_ids   = [i for t,i in vocab.items() if t.isdigit()]
   quote_ids   = [i for t,i in vocab.items() if '"' in t]
   plain_ids   = [i for t,i in vocab.items() if '"' not in t]
   dot_id      = vocab.get(".")
   minus_id    = vocab.get("-")
   end_ids     = [encode(c)[0] for c in (",", "}", " ", "\\n")]

2. pick(allowed) (test_03)
   logits = model(ids)
   masked = np.full(len(logits), -inf)
   masked[allowed] = logits[allowed]
   return int(np.argmax(masked))

3. choose(options) (test_08, test_09)
   paths = {option: encode(option) for option in options}
   while more than one option remains:
       allowed = next token ids of the remaining options
       token = forced if 1 allowed, else pick(allowed)
       drop options that don't match; append token
   append the winner's leftover tokens; return it

4. gen_string() (test_06)
   loop:
       logits = model(ids)
       closing = best token containing '"'
       content = best token without '"'
       if closing wins: keep any text before the quote; break
       else: append content token

5. gen_number(integer_only) (test_07)
   loop:
       allowed = digit_ids
       if no text yet: add minus_id
       if a digit was seen: add end_ids (+ dot_id once, floats only)
       token = pick(allowed)
       if token in end_ids: break
       append token

6. run(prompt) — puts it all together
   ids = encode(header + '{"name": "')
   name = choose(function names)
   add('", "parameters": {')
   for key, schema in function.parameters:
       add('"key": ')
       if string:  add('"'); value = gen_string(); add('"')
       if boolean: value = choose(["true", "false"]) == "true"
       if number:  value = gen_number(integer_only=schema.type=="integer")
       if not last: add(", ")
   add("}}")
   return {"prompt": prompt, "name": name, "parameters": parameters}
""")
