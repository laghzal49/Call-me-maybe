from typing import Dict, List, Optional


class TrieNode:
    """One node of a token trie: children keyed by token id, plus a
    value."""

    def __init__(self) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.value: Optional[str] = None


class Trie:
    """A trie over token-id sequences, used to constrain decoding to a
    fixed set of options."""

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()

    def insert(self, ids: List[int], value: str) -> None:
        """Insert one token-id path ending in `value`."""
        node = self.root
        for token_id in ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.value = value
