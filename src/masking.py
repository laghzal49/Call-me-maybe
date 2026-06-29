"""Logit masking: choose one token id from the valid ones.

Picking the max over a subset is the same as setting every other token's logit
to negative infinity and then taking the argmax, which is exactly what the
subject describes as constrained decoding.
"""

from typing import List, Set


def pick_allowed(logits: List[float], allowed: Set[int]) -> int:
    """Highest-logit token among `allowed` (mask by inclusion).

    Used when the valid set is small: trie children, or numeric tokens.
    """
    return max(allowed, key=lambda i: logits[i])


def pick_excluding(logits: List[float], forbidden: Set[int]) -> int:
    """Highest-logit token that is not in `forbidden` (mask by exclusion).

    Used for strings, where nearly every token is valid except a few (the
    tokens that contain a double-quote).
    """
    best_id, best_val = -1, float("-inf")
    for i, val in enumerate(logits):
        if i in forbidden:
            continue
        if val > best_val:
            best_id, best_val = i, val
    return best_id
