from typing import List, Dict, Set
import re
import json
import string
from llm_sdk import Small_LLM_Model


def _has_unescaped_quote(s: str) -> bool:
    """Return True if string contains an unescaped double quote."""
    return bool(re.search(r'(?<!\\)(?:\\\\)*"', s))


class Vocab:
    """Classifies vocabulary tokens for constrained decoding."""

    def __init__(self, llm: Small_LLM_Model) -> None:
        self.llm = llm
        with open(llm.get_path_to_vocab_file(), encoding="utf-8") as f:
            self.token_to_id: Dict[str, int] = json.load(f)

        self.digit_ids: Set[int] = {
            tid for tok, tid in self.token_to_id.items()
            if (s := tok.lstrip("Ġ "))
            and all(c in string.digits + ".-" for c in s)
        }
        self.non_quote_ids: List[int] = []
        self.quote_ids: List[int] = []
        for tok, tid in self.token_to_id.items():
            if _has_unescaped_quote(tok):
                self.quote_ids.append(tid)
            else:
                self.non_quote_ids.append(tid)

    def find_char_token(self, char: str) -> int:
        """Return the token id for a single character."""
        if char in self.token_to_id:
            return self.token_to_id[char]
        for tok, tid in self.token_to_id.items():
            if tok.lstrip("Ġ ") == char:
                return tid
        raise ValueError(f"Error: no token found for {char!r}")
