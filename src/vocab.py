import json
from typing import Dict, List

from llm_sdk import Small_LLM_Model


class Vocab:
    """Wraps the model's tokenizer: encode/decode (with caching) plus
    the token-id groups derived from the vocab file, used to mask
    logits during constrained decoding."""

    def __init__(self, llm: Small_LLM_Model) -> None:
        self.llm = llm
        self._encode_cache: Dict[str, List[int]] = {}

        path = llm.get_path_to_vocab_file()
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

        self.digit_ids = [
            i for tok, i in vocab.items() if tok.isdigit()
        ]
        self.quote_ids = [i for tok, i in vocab.items() if '"' in tok]
        self.plain_ids = [
            i for tok, i in vocab.items() if '"' not in tok
        ]
        self.dot_id = vocab.get(".")
        self.minus_id = vocab.get("-")
        if not self.digit_ids:
            raise ValueError("Error: vocab has no digit tokens")
        if not self.quote_ids:
            raise ValueError(
                "Error: vocab has no quote-bearing tokens"
            )
        if not self.plain_ids:
            raise ValueError("Error: vocab has no non-quote tokens")

        self.end_ids = []
        for char in (",", "}", " ", "\n"):
            encoded = self.encode(char)
            if not encoded:
                raise ValueError(
                    f"Error: model tokenizer produced no token "
                    f"for {char!r}"
                )
            self.end_ids.append(encoded[0])

    def encode(self, text: str) -> List[int]:
        """Encode text to a flat list of token ids.

        Fixed JSON literals (braces, commas, key names, function
        names) repeat identically on every prompt, so results are
        cached to avoid redundant tokenizer calls (bonus: performance
        optimization).
        """
        cached = self._encode_cache.get(text)
        if cached is not None:
            return list(cached)
        ids: List[int] = self.llm.encode(text).tolist()[0]
        self._encode_cache[text] = ids
        return list(ids)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back to text."""
        return self.llm.decode(ids)
