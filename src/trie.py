"""Token-id trie used for fixed-choice constraints (names, booleans)."""

from typing import Dict, List, Optional

from llm_sdk import Small_LLM_Model


def encode_ids(llm: Small_LLM_Model, text: str) -> List[int]:
    """Turn text into a flat list of token ids (encode returns a 2-D tensor)."""
    ids: List[int] = llm.encode(text).tolist()[0]
    return ids


class TrieNode:
    """One node of a token-id trie."""

    def __init__(self) -> None:
        # child token id -> next node
        self.children: Dict[int, "TrieNode"] = {}
        # the full word, set only at the node that ends a word (else None)
        self.value: Optional[str] = None


class Trie:
    """A set of words stored as paths of token ids."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_ids: List[int], value: str) -> None:
        """Add one word, walking/creating one node per token id."""
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.value = value   # remember the word at its final node

    @classmethod
    def from_strings(cls, llm: Small_LLM_Model, values: List[str]) -> "Trie":
        """Build a trie from words by encoding each one to token ids."""
        trie = cls()
        for value in values:
            trie.insert(encode_ids(llm, value), value)
        return trie
