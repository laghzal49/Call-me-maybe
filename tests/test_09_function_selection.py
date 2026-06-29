"""
CONCEPT 9 — FUNCTION SELECTION (trie + model logits = semantic routing)
========================================================================
This is where constrained decoding becomes powerful.

The function name trie has ONE branching point — the step where two
function names first diverge in their token sequences.  At that step
the model's logits reflect which function is most relevant to the prompt.

Example:
  functions: ["fn_add_numbers", "fn_greet"]
  Suppose both encode as:
      fn_add_numbers → [10, 20, 30]
      fn_greet       → [10, 40]

  Trie structure:
    root
     └─[10]─▶ node_shared
               ├─[20]─▶ … ─▶ leaf "fn_add_numbers"
               └─[40]─▶ leaf "fn_greet"

  Step 1: token 10 — FORCED (only option). No model call.
  Step 2: tokens {20, 40} — BRANCHING. Model picks here.

  If prompt = "What is the sum of 3 and 5?" → model assigns high logit
  to token 20 (the "_add" direction) → picks fn_add_numbers.

  If prompt = "Say hello to Alice" → model assigns high logit to token 40
  → picks fn_greet.

The model is NOT asked to output the function name as text. It only
contributes a logit comparison at the branching point.

Run: uv run python tests/test_09_function_selection.py   (no model needed)
"""

import sys

sys.path.insert(0, ".")

from src.trie import Trie  # noqa: E402

# ── 1. build a function-name trie with fake tokens ────────────────────────────
# Two functions that share a prefix token (token 10 = "fn_"):
#   fn_add_numbers → [10, 20, 30]   ("fn_", "add_", "numbers")
#   fn_greet       → [10, 40]       ("fn_", "greet")
#   fn_sqrt        → [50, 60]       ("fn", "_sqrt")  — different first token

fn_trie = Trie()
fn_trie.insert([10, 20, 30], "fn_add_numbers")
fn_trie.insert([10, 40], "fn_greet")
fn_trie.insert([50, 60], "fn_sqrt")

root = fn_trie.root

# ── 2. verify the trie structure ──────────────────────────────────────────────
assert set(root.children.keys()) == {10, 50}
print("Root branches (first token options):", set(root.children.keys()))
print("  token 10 → 'fn_add_numbers' or 'fn_greet' (shared prefix)")
print("  token 50 → 'fn_sqrt'")

# The branching point for fn_add vs fn_greet is after token 10
node_10 = root.children[10]
assert set(node_10.children.keys()) == {20, 40}
print("\nAfter token 10: children =", set(node_10.children.keys()))
print("  THIS is where the model's logits decide the function")
print("  token 20 → fn_add_numbers   token 40 → fn_greet")

# ── 3. simulate two prompts ───────────────────────────────────────────────────


def walk_with_logit_preference(tok_preference_at_branch: int) -> str:
    """
    Walk the trie.  At branching points, pick tok_preference_at_branch
    if it is a valid child — simulating the model's argmax over logits.
    """
    node = root
    while node.children:
        if len(node.children) == 1:
            (tid, child), = node.children.items()
        else:
            tid = (
                tok_preference_at_branch
                if tok_preference_at_branch in node.children
                else next(iter(node.children))
            )
            child = node.children[tid]
        node = child
    return node.value or ""


# "What is the sum of 3 and 5?" → model assigns highest logit to token 20
result_add = walk_with_logit_preference(tok_preference_at_branch=20)
assert result_add == "fn_add_numbers"
print("\nPrompt: 'What is the sum of 3 and 5?'")
print("  model prefers token 20 →", result_add)

# "Say hello to Alice" → model assigns highest logit to token 40
result_greet = walk_with_logit_preference(tok_preference_at_branch=40)
assert result_greet == "fn_greet"
print("\nPrompt: 'Say hello to Alice'")
print("  model prefers token 40 →", result_greet)

# fn_sqrt has a different first token, so it wins immediately when chosen
result_sqrt = walk_with_logit_preference(tok_preference_at_branch=50)
assert result_sqrt == "fn_sqrt"
print("\nPrompt: 'What is the square root of 9?'")
print("  model prefers token 50 →", result_sqrt)

# ── 4. key insight ────────────────────────────────────────────────────────────
# We never feed the model a list of function names or ask it to "output the
# function name".  We just let it predict the next token, and the trie ensures
# that only token sequences that spell valid function names are possible.
# The model's semantic understanding is captured entirely in which logit is
# highest at the branching point.

print("\nOK: function selection uses model logits at exactly one branching point")
