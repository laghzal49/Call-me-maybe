"""
CONCEPT 8 — BOOLEAN DECODING (choose over "true" / "false")
============================================================
Booleans are just a special case of Decoder.choose().

We call choose(["true", "false"]).  The model walks the two token paths
step by step, constrained to whichever word is still reachable.  After
the walk, the result is either "true" or "false" and we convert:
result == "true"  →  Python True / False.

Why token paths instead of just comparing two logits?
  "true" and "false" may each tokenize into MULTIPLE tokens.
  We cannot compare them in a single step — we must pick token by token.
  choose() handles single-token and multi-token words uniformly.

Run: uv run python tests/test_08_boolean_decoding.py   (no model needed)
"""

import sys
from typing import Dict, List

sys.path.insert(0, ".")

# Fake token ids so no model is needed:
#  "true"  → tokens [100, 200]       ("tr" and "ue", for example)
#  "false" → tokens [300, 400, 500]  ("f", "al", "se")
ENCODED: Dict[str, List[int]] = {
    "true": [100, 200],
    "false": [300, 400, 500],
}

# ── 1. the only real choice is the first token ────────────────────────────────
first_allowed = sorted({ids[0] for ids in ENCODED.values()})
assert first_allowed == [100, 300]
print("Step 0 allowed tokens:", first_allowed)
print("  token 100 → leads to 'true'")
print("  token 300 → leads to 'false'")
print("  After that first token only ONE word remains → forced steps")

# ── 2. simulate the choose() loop ─────────────────────────────────────────────


def choose_bool(preferred_first_token: int) -> bool:
    """
    Same loop as Decoder.choose(["true", "false"]), with the model
    replaced by a stub that picks preferred_first_token at the branch.
    Returns the Python bool.
    """
    paths = dict(ENCODED)
    step = 0
    while len(paths) > 1:
        allowed = sorted({ids[step] for ids in paths.values()})
        if len(allowed) == 1:
            token = allowed[0]  # forced: no model call
        else:
            token = preferred_first_token  # the model picks here
        paths = {o: ids for o, ids in paths.items() if ids[step] == token}
        step += 1
    (choice,) = paths
    return choice == "true"


result_true = choose_bool(preferred_first_token=100)
assert result_true is True
print("\nchoose (model picks token 100) →", result_true)

result_false = choose_bool(preferred_first_token=300)
assert result_false is False
print("choose (model picks token 300) →", result_false)

# ── 3. key point ──────────────────────────────────────────────────────────────
# The model's logits at the first divergence decide "true" vs "false" —
# exactly like function-name selection, just with two fixed words.

print("\nOK: boolean decoding is choose() over two fixed words")
