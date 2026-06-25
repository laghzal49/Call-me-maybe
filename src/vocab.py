import re
from typing import Dict, Set, List
import string
import json
from llm_sdk import Small_LLM_Model


def has_unescaped_quote(s: str) -> bool:
    """Return True if string contains an unescaped double quote."""
    return bool(re.search(r'(?<!\\)(?:\\\\)*"', s))


class Vocab:
    """Loads vocab.json and classifies tokens.

    Classified by character set for constrained decoding.
    """

    def __init__(self, llm: Small_LLM_Model) -> None:
        self.llm = llm
        self.token_to_id: Dict[str, int] = self._load_vocab()
        self.digite_token_to_id: Set[int] = self._classify(
            string.digits + ".-"
        )
        self.quote_token_id: int = self._find_single_char_token('"')
        self.non_quote_ids: List[int] = []
        self.quote_ids: List[int] = []
        self._classify_quotes()

    def _load_vocab(self) -> Dict[str, int]:
        """Load and parse the vocabulary file."""
        with open(self.llm.get_path_to_vocab_file(), encoding="utf-8") as f:
            res: Dict[str, int] = json.load(f)
            return res

    def _classify_quotes(self) -> None:
        """Classify vocabulary tokens into quote and non-quote lists."""
        for token_str, tid in self.token_to_id.items():
            # Convert bytes to string if key is bytes representation
            if isinstance(token_str, bytes):
                try:
                    token_str = token_str.decode("utf-8", errors="ignore")
                except Exception:
                    pass
            # Exclude any tokens containing a double quote character to
            # prevent escaping JSON boundaries.
            if has_unescaped_quote(token_str):
                self.quote_ids.append(tid)
            else:
                self.non_quote_ids.append(tid)

    def _classify(self, allowed_chars: str) -> Set[int]:
        """Return ids of tokens made only of allowed_chars."""
        ids: Set[int] = set()
        for token_str, token_id in self.token_to_id.items():
            # Strip standard space indicators
            stripped = token_str.lstrip("Ġ ")
            if stripped and all(c in allowed_chars for c in stripped):
                ids.add(token_id)
        return ids

    def _find_single_char_token(self, char: str) -> int:
        """Return the id of the token whose text is exactly `char`."""
        for token_str, token_id in self.token_to_id.items():
            if token_str == char:
                return token_id
        # Fallback to stripped matching
        for token_str, token_id in self.token_to_id.items():
            if token_str.lstrip("Ġ ") == char:
                return token_id
        raise ValueError(f"Error: no token found for character {char!r}")
