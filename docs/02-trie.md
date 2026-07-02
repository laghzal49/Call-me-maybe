# 02 — `src/trie.py`

## Role

A **trie** (prefix tree) over token-id sequences, used to constrain choices that
must be one of a small fixed set of words: a function name, or `true`/`false`.

## Theory

When the answer must be one of a few known strings, we encode each string into
its token ids and store them as paths in a tree. At any point during generation,
the **only valid next tokens are the children of the current node**. Walking the
tree therefore makes invalid words impossible while still letting the model
choose *which* valid word, by picking among the children.

Example for `true` and `false` (illustrative ids):

```
root ─┬─ [t][r][u][e]         -> value "true"
      └─ [f][a][l][s][e]      -> value "false"
```

At the root there are two children (`t`, `f`): a real choice. After that, each
path is forced (one child only).

## Inputs / Outputs

- `encode_ids(llm, text) -> List[int]` — text to a flat list of token ids.
- `Trie.from_strings(llm, ["true", "false"]) -> Trie` — a ready trie.
- During generation: a `TrieNode` exposes `.children` (id → node) and `.value`
  (the full word, set only at a leaf).

## How it works

- `encode_ids` calls `llm.encode(text)`, which returns a 2-D tensor
  `[[id, id, ...]]`; we take row `0` with `.tolist()[0]`.
- `TrieNode` holds `children: Dict[int, TrieNode]` and `value: Optional[str]`.
- `Trie.insert(token_ids, value)` walks the tree, creating a node per token id,
  and stores `value` on the final node so we can read the word back later.
- `Trie.from_strings(llm, values)` encodes each word and inserts it.

## Why this design

- **Why token ids, not characters?** The model predicts *tokens*, so the
  constraint must be expressed in tokens. A character trie would not line up with
  what the model can emit.
- **Why store `value` on the leaf?** After walking, we want the original string
  (e.g. `"fn_greet"`) without re-decoding the tokens — the leaf just hands it back.
- **Why a class and not a list of candidates?** A trie naturally handles **shared
  prefixes** (all function names start with `fn_`): the shared part is a single
  chain of forced nodes, and branching happens only where the words differ. The
  decoder's `_walk` uses exactly that to skip model calls on the forced part.

## How to reimplement

1. Write `encode_ids` to flatten the SDK's 2-D encode output.
2. Make a `TrieNode` with a `children` dict and an optional `value`.
3. Make a `Trie` with `insert(ids, value)` that walks/creates nodes and marks the
   leaf, plus a `from_strings` classmethod that encodes and inserts each word.

## Edge cases

- If one valid word is a token-prefix of another, the shorter word's node is both
  a leaf (`value` set) **and** has children. The decoder's `_walk` keeps
  descending while children exist; our function names are not prefixes of each
  other, so this does not occur in practice.
