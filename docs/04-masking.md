# 04 — `src/masking.py`

## Role

Turn a list of logits into the **chosen token id**, considering only the valid
tokens. This is the literal "constrained decoding" step.

## Theory

The subject defines constrained decoding as: *the model produces logits for all
tokens; you set the logits of invalid tokens to negative infinity; you sample
only from the rest.* We use greedy decoding (take the most likely token), so
"sample from the valid set" becomes "**take the max logit among the valid ids**".

Taking the max over a subset is mathematically identical to setting every other
logit to `-inf` and then taking the global argmax — we just skip building the
`-inf` array, which is faster and clearer.

## Inputs / Outputs

- `pick_allowed(logits, allowed) -> int` — highest-logit id **in** `allowed`.
- `pick_excluding(logits, forbidden) -> int` — highest-logit id **not in**
  `forbidden`.
- Input `logits` is the `List[float]` returned by the SDK (one per token id).

## How it works

- `pick_allowed`: `max(allowed, key=lambda i: logits[i])`. Used when the valid set
  is small — trie children, or the numeric token set.
- `pick_excluding`: scan all logits once, skip ids in `forbidden`, track the best.
  Used for strings, where almost every token is valid except the few that contain
  a quote, so listing the *allowed* set (≈150k ids) would be wasteful.

## Why this design

- **Two functions, by inclusion vs. exclusion.** When the valid set is tiny, loop
  over it (`pick_allowed`). When the *invalid* set is tiny, loop over everything
  and skip the invalid ones (`pick_excluding`). Each picks the cheaper loop.
- **Why greedy (argmax) and not sampling?** Determinism and reliability. For
  function calling we want the single most likely valid token, not variety.
- **Why a separate file?** This is the heart of "constrained decoding" and is pure
  (no model, no state). Isolating it makes the concept easy to find and test.

## How to reimplement

1. `pick_allowed`: return the element of `allowed` with the largest `logits[i]`.
2. `pick_excluding`: iterate `enumerate(logits)`, skip ids in `forbidden`, keep the
   index of the largest value seen.

## Edge cases

- `pick_allowed` with an empty `allowed` raises (a bug in the caller's constraint);
  callers always pass a non-empty set.
- `pick_excluding` returns `-1` only if every token is forbidden, which never
  happens (we forbid only quote-bearing tokens).
