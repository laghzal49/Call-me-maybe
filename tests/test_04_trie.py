"""
CONCEPT 4 — TRIE (prefix tree over token ids)
===============================================
Problem: function names like "fn_add_numbers" span MULTIPLE tokens.
We need to constrain the model to generate only valid function names,
one token at a time.

Solution: a TRIE — a tree where each edge is one token id.
  - Every path from root to leaf spells out a valid function name.
  - At each generation step, node.children.keys() = the set of valid
    NEXT token ids.  Feed that set directly to _pick().

Two types of steps:
  FORCED step   (1 child):  only one valid next token — skip the model.
  BRANCHING step (>1 child): real choice — call _pick(logits, children).

At branching points the model's logits determine which branch to take.
That is where the model's "intelligence" about the prompt is used.

Example trie for ["fn_add", "fn_greet"]:

  root
   └─[tok_fn]─▶ node
                 ├─[tok__add]──▶ leaf  value="fn_add"
                 └─[tok_greet]─▶ leaf  value="fn_greet"

token for "fn" is shared → FORCED step.
At the branch, the model picks between "_add" and "greet" tokens.

Run: uv run python tests/test_04_trie.py   (no model needed)
"""

import sys

sys.path.insert(0, ".")

from src.trie import Trie, TrieNode  # noqa: E402

# ── 1. build a trie manually with fake token ids ──────────────────────────────
# In real use, token ids come from llm.encode(word).tolist()[0].
# Here we assign them by hand so the test runs without the model.

trie = Trie()
#               token path     word
trie.insert([10, 20, 30],    "cat")
trie.insert([10, 20, 40],    "car")   # "cat" and "car" share [10, 20]
trie.insert([50, 60],        "dog")

root = trie.root

# ── 2. inspect the trie structure ─────────────────────────────────────────────

# Root: two possible first tokens — 10 (cat/car) or 50 (dog)
assert set(root.children.keys()) == {10, 50}
print("Root children (valid first tokens):", set(root.children.keys()))
print("  token 10 → leads to 'cat' or 'car' (shared prefix)")
print("  token 50 → leads to 'dog'")

# After token 10: only token 20 is valid → FORCED (skip model)
node_10 = root.children[10]
assert set(node_10.children.keys()) == {20}
print("\nAfter token 10: children =", set(node_10.children.keys()))
print("  Only one choice → FORCED step, no model call needed")

# After token 10 → 20: tokens 30 or 40 → BRANCHING (call model)
node_10_20 = node_10.children[20]
assert set(node_10_20.children.keys()) == {30, 40}
print("\nAfter tokens [10, 20]: children =", set(node_10_20.children.keys()))
print("  Two choices → BRANCHING step, model picks here")
print("  token 30 → 'cat'   token 40 → 'car'")

# Leaves store the complete word
leaf_cat = node_10_20.children[30]
assert leaf_cat.value == "cat"
assert leaf_cat.children == {}   # leaf has no children
print("\nLeaf (token 30): value =", repr(leaf_cat.value))
print("  No children → trie walk ends here")

# ── 3. simulate a trie walk ───────────────────────────────────────────────────
# _walk() in decoder.py does exactly this.


def walk(root: TrieNode, preferred_at_branch: int) -> str:
    """Walk the trie; at branch points pick `preferred_at_branch` if available."""
    node = root
    while node.children:
        if len(node.children) == 1:
            (tid, child), = node.children.items()   # forced: only one option
            print(f"  forced  → token {tid}")
        else:
            # In real code: tid = _pick(logits, set(node.children))
            tid = (
                preferred_at_branch
                if preferred_at_branch in node.children
                else next(iter(node.children))
            )
            child = node.children[tid]
            print(f"  branch  → model picks token {tid}")
        node = child
    return node.value or ""


print("\nWalk (model prefers token 40 at branch):")
result = walk(trie.root, preferred_at_branch=40)
assert result == "car"
print(f"  result: {result!r}")

print("\nWalk (model prefers token 30 at branch):")
result = walk(trie.root, preferred_at_branch=30)
assert result == "cat"
print(f"  result: {result!r}")

print("\nOK: trie correctly constrains multi-token word generation")
