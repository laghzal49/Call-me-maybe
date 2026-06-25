from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class TrieNode(BaseModel):
    """Single node in the token_id trie"""

    children: Dict[int, "TrieNode"] = Field(default_factory=dict)
    value: Optional[str] = None


TrieNode.model_rebuild()


class Trie(BaseModel):
    """Trie over token-id sequences, used to constrain decoding to a closed set
    of valid strings (function names, booleans, etc.)."""

    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_ids: List[int], value: str) -> None:
        """Insert one token-id sequence, tagging final node with `value`."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.value = value

    def allowed_next_ids(self, node: TrieNode) -> List[int]:
        """Token ids that keep at least one path alive from `node`."""
        return list(node.children.keys())

    def step(self, node: TrieNode, token_id: int) -> TrieNode:
        if token_id not in node.children:
            raise ValueError(
                f"Error: token id {token_id} is not a valid path here"
            )
        return node.children[token_id]
