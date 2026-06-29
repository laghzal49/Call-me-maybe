"""
CONCEPT 2 — LOGITS
===================
At every generation step the LLM outputs ONE FLOAT PER TOKEN in the
vocabulary.  These floats are called LOGITS.

    vocabulary size = ~150 000 tokens  →  150 000 logits per step

Higher logit = the model thinks that token is more likely next.

GREEDY decoding: pick the token with the highest logit (argmax).
CONSTRAINED decoding: pick the highest logit AMONG VALID TOKENS ONLY.

The model is never changed.  We only filter which logits we look at.

Run: uv run python tests/test_02_logits.py   (no model needed)
"""

import numpy as np

# ── 1. what logits look like ─────────────────────────────────────────────────
# Imagine a tiny vocabulary of 5 tokens.

token_names = ["5", "a", "3", " ", ","]
logits = [2.1, 4.8, 3.2, 1.0, 0.5]
#          ^     ^    ^    ^    ^
# token:   0     1    2    3    4

print("Token vocabulary (simplified):")
for i, (name, logit) in enumerate(zip(token_names, logits)):
    print(f"  id={i}  text={name!r:4s}  logit={logit}")

# ── 2. greedy decoding ───────────────────────────────────────────────────────
# Without any constraint, the model just picks the highest logit.

arr = np.array(logits)
greedy = int(np.argmax(arr))

print(f"\nGreedy argmax → token {greedy} ({token_names[greedy]!r})")
print("Problem: the model wants 'a', but we need a digit for a number field!")

# ── 3. the logit tells us the model's preference — not a rule ────────────────
# The model learned from text, so it knows 'a' often follows many things.
# But for our JSON schema, the value at this position MUST be a digit.
# We need to OVERRIDE the model's preference with our schema constraint.

# ── 4. what argmax over allowed tokens looks like ────────────────────────────
# If we only allow digit tokens {0, 2} (tokens for "5" and "3"):

allowed = {0, 2}
best_allowed = max(allowed, key=lambda i: logits[i])

print(f"\nAllowed digit tokens: {allowed}")
print(f"Best among allowed  → token {best_allowed} ({token_names[best_allowed]!r})")
print("Now the model picks '3' — still uses the model's judgement,")
print("but only among tokens that keep the output valid.")

# ── 5. the key insight ────────────────────────────────────────────────────────
# The model still runs. The logits still come from the model's knowledge.
# We just tell it: "of all the tokens you considered, pick the best VALID one."
# This is not heuristics — the function choice still comes from the model.

assert best_allowed == 2   # token 2 ("3") has logit 3.2 > token 0 ("5") 2.1
print("\nOK: constrained pick uses model knowledge, enforces schema correctness")
