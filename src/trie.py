"""A token-id trie used to constrain the model's choice among a
fixed set of string options (e.g. function names, "true"/"false")
during decoding."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TrieNode(BaseModel):
    """One node of a token trie: children keyed by token id, plus a
    value.

    Attributes:
        children: Child nodes keyed by the next token id.
        value: The option string this node completes, if any.
        end: Whether this node is the end of at least one inserted
            option (a closing quote may legally follow here).
    """

    children: Dict[int, "TrieNode"] = Field(default_factory=dict)
    value: Optional[str] = None
    end: bool = False


class Trie(BaseModel):
    """A trie over token-id sequences, used to constrain decoding to a
    fixed set of options.

    Attributes:
        root: The trie's root node.
    """

    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, ids: List[int], value: str) -> None:
        """Insert one token-id path ending in `value`.

        Args:
            ids: The sequence of token ids that spell out `value`.
            value: The option string this path represents.

        Returns:
            None.
        """
        node = self.root
        for token_id in ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.value = value
        node.end = True
