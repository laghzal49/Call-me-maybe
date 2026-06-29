"""
CONCEPT 3 — CONSTRAINED DECODING (the -inf trick)
===================================================
This is the CORE idea of the entire project.

Instead of letting argmax run over all 150 000 tokens, we:
  1. Start with the full logit array from the LLM.
  2. Set every INVALID token's logit to -infinity.
  3. Run argmax — it can only pick from what's left.

    masked = np.full(len(logits), -inf)   # start: everything forbidden
    masked[allowed_ids] = logits[allowed_ids]   # lift the valid ones
    best = argmax(masked)                  # only valid tokens compete

This is what _pick() does in decoder.py.

Why -inf and not 0?
  Logits can be negative.  Setting forbidden tokens to 0 might still
  let them win if all valid logits are also negative.  -inf guarantees
  they can NEVER win regardless of scale.

Run: uv run python tests/test_03_constrained_decoding.py   (no model needed)
"""

import numpy as np

# ── 1. the _pick() function — the whole algorithm in 4 lines ─────────────────


def pick(logits: list, allowed: set) -> int:
    """Constrained argmax: valid tokens compete, invalid tokens are -inf."""
    arr = np.full(len(logits), -np.inf)   # everything starts at -inf
    idx = list(allowed)
    arr[idx] = np.array(logits)[idx]      # copy in only the allowed logits
    return int(np.argmax(arr))            # argmax can only reach valid tokens


# ── 2. demo with digit constraint ────────────────────────────────────────────

logits = [2.1, 4.8, 3.2, 1.0, 0.5]
#   id:   0     1    2    3    4
# text:  "5"   "a"  "3"  " "  ","

digit_ids = {0, 2}    # tokens whose text is all digits: "5" and "3"

result = pick(logits, digit_ids)
assert result == 2    # "3" wins (logit 3.2) over "5" (logit 2.1)

print("Logits:           ", logits)
print("Allowed (digits): ", digit_ids)
print("Winner:           ", result, "(token 'a' had logit 4.8 but was forbidden)")

# ── 3. exclusion variant — used for strings ───────────────────────────────────
# For strings we have the OPPOSITE situation: almost everything is valid
# EXCEPT tokens that contain a quote character.
# Setting forbidden tokens to -inf is more efficient than listing 149 000 valid ones.


def pick_except(logits: list, forbidden: set) -> int:
    """Constrained argmax: forbidden tokens are set to -inf."""
    arr = np.array(logits, dtype=float)
    arr[list(forbidden)] = -np.inf    # mask out only the forbidden ones
    return int(np.argmax(arr))


quote_ids = {3, 4}   # fake: tokens whose text contains "
best_content = pick_except(logits, quote_ids)
# tokens 3 and 4 are masked, so best among {0,1,2} wins → token 1 ("a", logit 4.8)
assert best_content == 1
print("\nExclusion (mask quotes):")
print("  Forbidden:", quote_ids)
print("  Winner:   ", best_content, "(token 'a' — no longer competing with quotes)")

# ── 4. why this is not heuristics ────────────────────────────────────────────
# The logits still come from the model. The model still "decides" based
# on its training. We only restrict WHICH TOKENS it may decide between.
# The function name selection works the same way — the model's logits
# determine which function is most relevant to the prompt.

print("\nKey point: logits come from the model, constraints come from the schema.")
print("The model provides intelligence; the code provides correctness.")
