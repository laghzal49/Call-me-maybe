import json
import re
import string
from typing import Dict, List, Set, Tuple

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, Field


def _has_unescaped_quote(value: str) -> bool:
    """Return True if string contains an unescaped double quote."""
    return bool(re.search(r'(?<!\\)(?:\\\\)*"', value))


class Vocab(BaseModel):
    """Classifies vocabulary tokens for constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    token_to_id: Dict[str, int] = Field(default_factory=dict)
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    digit_ids: Set[int] = Field(default_factory=set)
    numeric_tokens: List[Tuple[int, str]] = Field(default_factory=list)
    non_quote_ids: List[int] = Field(default_factory=list)
    quote_ids: List[int] = Field(default_factory=list)
    quote_id_set: Set[int] = Field(default_factory=set)
    string_ids: List[int] = Field(default_factory=list)
    number_stop_ids: List[int] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        """Load and classify the model vocabulary after validation."""
        with open(self.llm.get_path_to_vocab_file(), encoding="utf-8") as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict):
            raise ValueError("Error: model vocabulary must be a JSON object")
        self.token_to_id = {
            str(token): int(token_id)
            for token, token_id in loaded.items()
        }
        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

        self.quote_ids = []
        self.non_quote_ids = []
        self.digit_ids = set()
        self.numeric_tokens = []
        for token, token_id in self.token_to_id.items():
            normalized = token.lstrip("Ġ ")
            if normalized and all(
                char in string.digits + ".-" for char in normalized
            ):
                self.digit_ids.add(token_id)
                self.numeric_tokens.append((token_id, normalized))
            if _has_unescaped_quote(token):
                self.quote_ids.append(token_id)
            else:
                self.non_quote_ids.append(token_id)
        self.quote_id_set = set(self.quote_ids)
        self.string_ids = self.non_quote_ids + self.quote_ids
        self.number_stop_ids = self._find_existing_tokens(
            [",", "}", "\n", " ", '"']
        )

    def find_char_token(self, char: str) -> int:
        """Return the token id for a single character."""
        if char in self.token_to_id:
            return self.token_to_id[char]
        for token, token_id in self.token_to_id.items():
            if token.lstrip("Ġ ") == char:
                return token_id
        raise ValueError(f"Error: no token found for {char!r}")

    def _find_existing_tokens(self, chars: List[str]) -> List[int]:
        """Return token ids for chars that exist in the vocabulary."""
        token_ids: List[int] = []
        for char in chars:
            try:
                token_ids.append(self.find_char_token(char))
            except ValueError:
                pass
        return token_ids
