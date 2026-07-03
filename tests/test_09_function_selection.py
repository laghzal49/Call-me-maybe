"""
CONCEPT 9 — FUNCTION SELECTION (path filtering + model logits)
===============================================================
This is where constrained decoding becomes powerful.

Decoder.choose() encodes every function name to its token ids and keeps
a dict of "still possible" options.  At each step the allowed tokens are
the next ids of the remaining options.  If only one token is allowed the
step is forced (no model call).  If several are allowed, the model's
logits pick — and options that don't match the picked token are dropped.

Example:
  functions: ["fn_add_numbers", "fn_greet", "fn_sqrt"]
  Suppose they encode as:
      fn_add_numbers → [10, 20, 30]
      fn_greet       → [10, 40]
      fn_sqrt        → [50, 60]

  Step 0: allowed = {10, 50} — BRANCHING.  Model picks.
  If it picks 10: fn_sqrt is dropped.
  Step 1: allowed = {20, 40} — BRANCHING.  Model picks again.
  If it picks 20: only fn_add_numbers remains → done.

The model is NOT asked to output the function name as text.  It only
contributes a logit comparison where valid names diverge.

Run: uv run python tests/test_09_function_selection.py   (no model needed)
"""

import sys
from typing import Dict, List

sys.path.insert(0, ".")

# Fake token ids for three function names:
#   fn_add_numbers → [10, 20, 30]   ("fn_", "add_", "numbers")
#   fn_greet       → [10, 40]       ("fn_", "greet")
#   fn_sqrt        → [50, 60]       ("fn", "_sqrt")  — different first token
ENCODED: Dict[str, List[int]] = {
    "fn_add_numbers": [10, 20, 30],
    "fn_greet": [10, 40],
    "fn_sqrt": [50, 60],
}


def choose(preferred: int) -> str:
    """
    Same loop as Decoder.choose(), with the model replaced by a stub:
    at branching points, pick `preferred` if allowed, else the first
    allowed token — simulating argmax over logits.
    """
    paths = dict(ENCODED)
    step = 0
    while len(paths) > 1:
        allowed = sorted({ids[step] for ids in paths.values()})
        if len(allowed) == 1:
            token = allowed[0]  # forced: no model call
        else:
            token = preferred if preferred in allowed else allowed[0]
        paths = {o: ids for o, ids in paths.items() if ids[step] == token}
        step += 1
    (choice,) = paths
    return choice


# ── 1. the first step is a real branch ────────────────────────────────────────
first_allowed = sorted({ids[0] for ids in ENCODED.values()})
assert first_allowed == [10, 50]
print("Step 0 allowed tokens:", first_allowed)
print("  token 10 → 'fn_add_numbers' or 'fn_greet' (shared prefix)")
print("  token 50 → 'fn_sqrt'")

# ── 2. simulate three prompts ─────────────────────────────────────────────────
# "What is the sum of 3 and 5?" → model assigns highest logit to token 20
result_add = choose(preferred=20)
assert result_add == "fn_add_numbers"
print("\nPrompt: 'What is the sum of 3 and 5?'")
print("  model prefers token 20 →", result_add)

# "Say hello to Alice" → model assigns highest logit to token 40
result_greet = choose(preferred=40)
assert result_greet == "fn_greet"
print("\nPrompt: 'Say hello to Alice'")
print("  model prefers token 40 →", result_greet)

# fn_sqrt has a different first token, so it wins at step 0 when chosen
result_sqrt = choose(preferred=50)
assert result_sqrt == "fn_sqrt"
print("\nPrompt: 'What is the square root of 9?'")
print("  model prefers token 50 →", result_sqrt)

# ── 3. key insight ────────────────────────────────────────────────────────────
# We never ask the model to "output the function name".  We just let it
# predict the next token among the ids that still spell a valid name.
# Its semantic understanding is captured entirely in which allowed logit
# is highest where the names diverge.

print("\nOK: function selection uses model logits only where names diverge")
