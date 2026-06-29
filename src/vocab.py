"""Token sets used to mask logits during decoding."""

import json
from typing import Dict, Set

from llm_sdk import Small_LLM_Model

_NUMBER_END_CHARS = [",", "}", "]", " ", "\n"]


class Vocab:
    """Token id <-> text map plus the masking sets for numbers/strings."""

    def __init__(self, llm: Small_LLM_Model) -> None:
        """Load the vocab and precompute every reusable token set once."""
        self.llm: Small_LLM_Model = llm
        self.text_to_id: Dict[str, int] = self._load_json()
        self.id_to_text: Dict[int, str] = {i: t for t, i in self.text_to_id.items()}
        self.digit_ids: Set[int] = {
            i
            for t, i in self.text_to_id.items()
            if t and all(c in "0123456789" for c in t)
        }
        self.dot_id: int = self.text_to_id.get(".", -1)
        self.minus_id: int = self.text_to_id.get("-", -1)
        self.number_end_ids: Set[int] = self._encode_first(_NUMBER_END_CHARS)
        # every token whose text contains a double-quote. These are the
        # candidates for CLOSING a string (e.g. '"', '",', '"}'). We check
        # startswith, not 'in', so that tokens like '\"' (backslash-quote,
        # valid inside a JSON string / regex pattern) are allowed as content.
        self.string_quote_ids: Set[int] = {
            i for t, i in self.text_to_id.items() if t.startswith('"')
        }

    def _load_json(self) -> Dict[str, int]:
        """Load the vocab JSON file the SDK points us to."""
        vocab_path: str = self.llm.get_path_to_vocab_file()
        try:
            with open(vocab_path, encoding="utf-8") as file:
                data: Dict[str, int] = json.load(file)
                return data
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File Not Found: {vocab_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Error in JSON decode: {error}") from error

    def _encode_first(self, chars: list[str]) -> Set[int]:
        """Token id of the first token of each given character."""
        ids: Set[int] = set()
        for char in chars:
            encoded = self.llm.encode(char).tolist()[0]  # 2-D -> first row
            if encoded:
                ids.add(encoded[0])
        return ids

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to text."""
        text: str = self.llm.decode([token_id])
        return text

    def number_tokens(
        self, *, started: bool, has_digit: bool, has_dot: bool
    ) -> Set[int]:
        """Digits, a sign only as the first char, and one dot only after a digit."""
        allowed = set(self.digit_ids)
        if not started and self.minus_id != -1:  # sign only at the very start
            allowed.add(self.minus_id)
        if has_digit and not has_dot and self.dot_id != -1:  # one dot, after a digit
            allowed.add(self.dot_id)
        return allowed

    def integer_tokens(self, *, started: bool) -> Set[int]:
        """Like number_tokens but never allows a decimal point."""
        allowed = set(self.digit_ids)
        if not started and self.minus_id != -1:  # sign only at the very start
            allowed.add(self.minus_id)
        return allowed
