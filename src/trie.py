"""A token-id trie used to constrain the model's choice among a
fixed set of string options (e.g. function names, "true"/"false")
during decoding."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TrieNode(BaseModel):
    """One node of a token trie.

    Attributes:
        children: Child nodes keyed by the next token id.
        value: The option string this node completes, if any.
    """

    children: Dict[int, "TrieNode"] = Field(default_factory=dict)
    value: Optional[str] = None


class Trie(BaseModel):
    """A trie over token-id sequences, used to constrain decoding to a
    fixed set of options.

    Attributes:
        root: The trie's root node.
    """

    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, ids: List[int], value: str) -> None:
        """Insert one option into the trie.

        Args:
            ids: The token-id path encoding the option, in order.
            value: The option string stored on the path's final node.

        Returns:
            None.
        """
        node = self.root
        for token_id in ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.value = value
