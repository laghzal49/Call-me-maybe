"""Wraps a model's tokenizer with encode/decode helpers and the
token-id groups (digits, quotes, plain text, end-of-value markers)
that the Decoder uses to mask logits during constrained decoding."""

import json
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model


def _load_vocab(path: str) -> Dict[str, int]:
    """Load the token->id mapping from the model's vocab file.

    Args:
        path: Filesystem path to the model's vocab JSON file.

    Returns:
        The token->id mapping read from the file.

    Raises:
        ValueError: If the file cannot be read, is not valid JSON,
            or does not contain a non-empty JSON object.
    """
    try:
        with open(path, encoding="utf-8") as file:
            vocab = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Error: invalid vocab JSON: {error}") from error
    except Exception as error:
        raise ValueError(
            f"Error: cannot load vocab file: {error}"
        ) from error

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(
            "Error: vocab file must contain a non-empty token->id object"
        )
    if any(type(value) is not int or value < 0 for value in vocab.values()):
        raise ValueError("Error: vocab ids must be non-negative integers")
    return vocab


class Vocab(BaseModel):
    """Wraps the model's tokenizer: encode/decode (with caching) plus
    the token-id groups derived from the vocab file, used to mask
    logits during constrained decoding.

    Attributes:
        llm: The underlying model, used for encode/decode/logits.
        digit_ids: Token ids whose text is entirely digits.
        quote_ids: Token ids whose text contains a `"`.
        plain_ids: Token ids whose text contains no `"`.
        end_ids: Token ids for the characters that legally end a
            number (",", "}", " ", "\\n").
        dot_id: Token id for "." if present in the vocab, else None.
        minus_id: Token id for "-" if present in the vocab, else None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    digit_ids: List[int]
    quote_ids: List[int]
    plain_ids: List[int]
    end_ids: List[int]
    dot_id: Optional[int] = None
    minus_id: Optional[int] = None

    _encode_cache: Dict[str, List[int]] = PrivateAttr(default_factory=dict)

    def __init__(self, llm: Small_LLM_Model) -> None:
        """Derive the token-id groups from the model's vocab file.

        Args:
            llm: The loaded model whose tokenizer and vocab file this
                instance wraps.

        Raises:
            ValueError: If the vocab file is unusable, if it lacks
                the digit/quote/plain tokens the decoder needs, or if
                the tokenizer produces no token for one of the
                number-ending characters.
        """
        vocab = _load_vocab(llm.get_path_to_vocab_file())

        digit_ids = [i for token, i in vocab.items()
                     if token.isascii() and token.isdigit()]
        quote_ids = [i for token, i in vocab.items() if '"' in token]
        plain_ids = [i for token, i in vocab.items() if '"' not in token]
        if not digit_ids:
            raise ValueError("Error: vocab has no digit tokens")
        if not quote_ids:
            raise ValueError("Error: vocab has no quote-bearing tokens")
        if not plain_ids:
            raise ValueError("Error: vocab has no non-quote tokens")

        end_ids: List[int] = []
        for char in (",", "}", " ", "\n"):
            encoded = llm.encode(char).tolist()[0]
            if not encoded:
                raise ValueError(
                    f"Error: model tokenizer produced no token for {char!r}"
                )
            end_ids.append(encoded[0])

        super().__init__(
            llm=llm,
            digit_ids=digit_ids,
            quote_ids=quote_ids,
            plain_ids=plain_ids,
            end_ids=end_ids,
            dot_id=vocab.get("."),
            minus_id=vocab.get("-"),
        )

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids, caching per literal string.

        Args:
            text: The text to tokenize.

        Returns:
            A new list of token ids (safe for the caller to mutate).
        """
        cached = self._encode_cache.get(text)
        if cached is not None:
            return list(cached)
        ids: List[int] = self.llm.encode(text).tolist()[0]
        self._encode_cache[text] = ids
        return list(ids)

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text.

        Args:
            ids: The token ids to decode.

        Returns:
            The decoded text.
        """
        return self.llm.decode(ids)
