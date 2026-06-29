"""
CONCEPT 8 — BOOLEAN DECODING (trie over "true" / "false")
===========================================================
Booleans are just a special case of trie walking.

We build a trie over exactly two words: ["true", "false"].
The model then walks the trie token by token, constrained to whichever
of the two words is still reachable at each step.

After the walk, the leaf's value is either "true" or "false".
We convert:  result == "true"  →  Python True / False.

Why a trie instead of just comparing two logits?
  "true" and "false" may each tokenize into MULTIPLE tokens.
  We cannot compare them in a single step — we must walk token by token.
  The trie handles both single-token and multi-token words uniformly.

Run: uv run python tests/test_08_boolean_decoding.py   (no model needed)
"""

import sys

sys.path.insert(0, ".")

from src.trie import Trie  # noqa: E402

# ── 1. build the boolean trie with fake token ids ─────────────────────────────
# In real code: Trie.from_strings(llm, ["true", "false"])
# Here we assign ids manually so no model is needed.

#  "true"  → tokens [100, 200]   ("tr" and "ue", for example)
#  "false" → tokens [300, 400, 500]  ("f", "al", "se")

bool_trie = Trie()
bool_trie.insert([100, 200], "true")
bool_trie.insert([300, 400, 500], "false")

root = bool_trie.root

# ── 2. inspect the trie ───────────────────────────────────────────────────────

assert set(root.children.keys()) == {100, 300}
print("Boolean trie root children:", set(root.children.keys()))
print("  token 100 → leads to 'true'")
print("  token 300 → leads to 'false'")

# After 100 → only 200 remains (forced step)
node_true = root.children[100]
assert set(node_true.children.keys()) == {200}
print("\nAfter token 100: children =", set(node_true.children.keys()))
print("  Only one choice → forced step (skip model)")

# Leaf
leaf_true = node_true.children[200]
assert leaf_true.value == "true"
assert leaf_true.children == {}
print("Leaf: value =", repr(leaf_true.value))

# ── 3. simulate the walk ──────────────────────────────────────────────────────


def walk_bool(preferred_first_token: int) -> bool:
    """
    Simulate _walk(bool_trie.root).
    At the root (branching point), pick preferred_first_token.
    All subsequent steps are forced.
    Returns the Python bool.
    """
    node = root
    while node.children:
        if len(node.children) == 1:
            (tid, child), = node.children.items()
        else:
            # This is the only branching point — the model picks here.
            # We simulate by using preferred_first_token.
            tid = preferred_first_token
            child = node.children[tid]
        node = child
    return (node.value or "") == "true"


result_true = walk_bool(preferred_first_token=100)
assert result_true is True
print("\nWalk (model picks token 100) →", result_true)

result_false = walk_bool(preferred_first_token=300)
assert result_false is False
print("Walk (model picks token 300) →", result_false)

# ── 4. key point ──────────────────────────────────────────────────────────────
# The model's logits at the branching point determine whether it generates
# "true" or "false" — just like with function names.
# After that first branching token, all remaining tokens are forced.

print("\nOK: boolean decoding is just a 2-word trie walk")
