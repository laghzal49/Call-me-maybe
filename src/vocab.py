import json
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model


def _load_vocab(path: str) -> Dict[str, int]:
    """Load the token->id mapping from the model's vocab file."""
    try:
        with open(path, encoding="utf-8") as file:
            vocab = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Error: invalid vocab JSON: {error}"
        ) from error
    except Exception as error:
        raise ValueError(
            f"Error: cannot load vocab file: {error}"
        ) from error

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(
            "Error: vocab file must contain a non-empty "
            "token->id object"
        )
    return vocab


class Vocab(BaseModel):
    """Wraps the model's tokenizer: encode/decode (with caching) plus
    the token-id groups derived from the vocab file, used to mask
    logits during constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    digit_ids: List[int]
    quote_ids: List[int]
    plain_ids: List[int]
    end_ids: List[int]
    dot_id: Optional[int] = None
    minus_id: Optional[int] = None

    _encode_cache: Dict[str, List[int]] = PrivateAttr(
        default_factory=dict
    )

    def __init__(self, llm: Small_LLM_Model) -> None:
        """Load the vocab file, derive the token-id groups, and let
        pydantic validate them."""
        vocab = _load_vocab(llm.get_path_to_vocab_file())

        digit_ids = [
            i for tok, i in vocab.items() if tok.isdigit()
        ]
        quote_ids = [i for tok, i in vocab.items() if '"' in tok]
        plain_ids = [
            i for tok, i in vocab.items() if '"' not in tok
        ]
        if not digit_ids:
            raise ValueError("Error: vocab has no digit tokens")
        if not quote_ids:
            raise ValueError(
                "Error: vocab has no quote-bearing tokens"
            )
        if not plain_ids:
            raise ValueError("Error: vocab has no non-quote tokens")

        end_ids: List[int] = []
        for char in (",", "}", " ", "\n"):
            encoded = llm.encode(char).tolist()[0]
            if not encoded:
                raise ValueError(
                    f"Error: model tokenizer produced no token "
                    f"for {char!r}"
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
        """Encode text to a flat list of token ids."""
        cached = self._encode_cache.get(text)
        if cached is not None:
            return list(cached)
        ids: List[int] = self.llm.encode(text).tolist()[0]
        self._encode_cache[text] = ids
        return list(ids)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back to text."""
        return self.llm.decode(ids)
