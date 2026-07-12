from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TrieNode(BaseModel):
    """One node of a token trie: children keyed by token id, plus a
    value."""

    children: Dict[int, "TrieNode"] = Field(default_factory=dict)
    value: Optional[str] = None
    end: bool = False


class Trie(BaseModel):
    """A trie over token-id sequences, used to constrain decoding to a
    fixed set of options."""

    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, ids: List[int], value: str) -> None:
        """Insert one token-id path ending in `value`."""
        node = self.root
        for token_id in ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.value = value
        node.end = True
