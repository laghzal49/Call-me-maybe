"""Token-id trie used for fixed-choice constraints (function names, booleans).

A trie stores words as paths of token ids.  Each node represents "where we
are so far".  Its children map the NEXT valid token id → the next node.

Why a trie and not a plain list?
  The model picks one token at a time.  At each step we only need to know
  which token ids are still consistent with at least one valid word.
  That is exactly node.children.keys() — O(1) lookup per step.

Forced vs branching steps:
  1 child  → only one valid next token; skip the model (no logit call needed).
  >1 child → real choice; call the model and constrain to node.children.
"""

from typing import Dict, List, Optional

from llm_sdk import Small_LLM_Model


def encode_ids(llm: Small_LLM_Model, text: str) -> List[int]:
    """Encode text to a flat list of token ids.

    llm.encode() returns a 2-D tensor shaped [1, n_tokens]; .tolist()[0]
    gives the inner list of integer ids.
    """
    ids: List[int] = llm.encode(text).tolist()[0]
    return ids


class TrieNode:
    """One node in the trie."""

    def __init__(self) -> None:
        # Maps next-token-id → child node.  Empty at leaves.
        self.children: Dict[int, "TrieNode"] = {}
        # Set only at the node that ENDS a word; None everywhere else.
        self.value: Optional[str] = None


class Trie:
    """A set of words stored as token-id paths from a shared root."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_ids: List[int], value: str) -> None:
        """Add one word by walking/creating one node per token id."""
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        # Store the complete word at the leaf so the caller knows which
        # word was selected after the walk finishes.
        node.value = value

    @classmethod
    def from_strings(cls, llm: Small_LLM_Model, values: List[str]) -> "Trie":
        """Build a trie by encoding each word to token ids, then inserting."""
        trie = cls()
        for value in values:
            trie.insert(encode_ids(llm, value), value)
        return trie
