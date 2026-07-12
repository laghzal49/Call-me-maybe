import json
import sys
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from llm_sdk import Small_LLM_Model
from src.parsing import FunctionDefinition, TypeSchema
from src.trie import Trie
from src.vocab import Vocab

JsonObject = Dict[str, Any]

MAX_STRING_TOKENS = 64
MAX_NUMBER_TOKENS = 24


def _unescaped_quote_index(text: str, chunk: str) -> Any:
    """Index of the first quote in `chunk` not escaped by a
    backslash (counting backslashes that end `text`), else None."""
    backslashes = len(text) - len(text.rstrip("\\"))
    for index, char in enumerate(chunk):
        if char == '"' and backslashes % 2 == 0:
            return index
        backslashes = backslashes + 1 if char == "\\" else 0
    return None


def _json_unescape(text: str) -> str:
    """Decode JSON string escapes (e.g. \\\\ -> \\); return the raw
    text if the model produced escapes JSON does not accept."""
    try:
        return str(json.loads(f'"{text}"'))
    except (json.JSONDecodeError, ValueError):
        return text


class Decoder(BaseModel):
    """Create once per run, then call run(prompt) for each prompt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    functions: Dict[str, FunctionDefinition]
    verbose: bool = False
    vocab: Vocab
    ids: List[int] = Field(default_factory=list)

    def __init__(
        self,
        llm: Small_LLM_Model,
        functions: Dict[str, FunctionDefinition],
        verbose: bool = False,
    ) -> None:
        """Build the vocab wrapper and let pydantic validate the
        fields."""
        super().__init__(
            llm=llm,
            functions=functions,
            verbose=verbose,
            vocab=Vocab(llm),
        )

    @property
    def block(self) -> str:
        """Render the function catalog as prompt text, one line per
        function: name, typed parameters, and description."""
        lines = []
        for fn in self.functions.values():
            params = []
            for k, v in fn.parameters.items():
                params.append(f"{k}: {v.type}")
            lines.append(
                f"- {fn.name}({', '.join(params)}): {fn.description}"
            )
        return "\n".join(lines)

    def add(self, text: str) -> None:
        """Append fixed JSON structure without asking the model."""
        self.ids += self.vocab.encode(text)

    def pick(self, allowed: List[int]) -> int:
        """One constrained decoding step: fetch logits, mask every
        token outside `allowed` to -inf, and return the argmax."""
        logits = np.array(
            self.llm.get_logits_from_input_ids(self.ids)
        )
        masked = np.full(len(logits), -np.inf)
        masked[allowed] = logits[allowed]
        token = int(np.argmax(masked))
        self.log(
            f"pick: {len(allowed)} allowed token(s) -> "
            f"chose id {token} "
            f"({self.vocab.decode([token])!r})"
        )
        return token

    def choose(self, options: List[str]) -> str:
        """Let the model pick one of `options` (e.g. a function name).

        Every option is encoded into a token trie; at each step only
        the tokens that continue at least one still-possible option
        are allowed.
        """
        if not options:
            raise ValueError(
                "Error: choose() needs at least one option"
            )

        trie = Trie()
        for option in options:
            trie.insert(self.vocab.encode(option), option)

        node = trie.root
        while node.value is None:
            if node.end is False:
                allowed = sorted(node.children)
            else:
                allowed = sorted(node.children) + self.vocab.quote_ids
            token = (
                allowed[0]
                if len(allowed) == 1
                else self.pick(allowed)
            )
            self.ids.append(token)
            node = node.children[token]
        return node.value

    def _emit(self, token: int) -> str:
        """Append one accepted token id and return its decoded
        text."""
        self.ids.append(token)
        text = self.vocab.decode([token])
        self.log(f"emit: id {token} decodes to {text!r}")
        return text

    def gen_string(self) -> str:
        """Generate a string; pick() over the full vocab stops on the
        first unescaped quote, then JSON escapes are decoded."""
        text = ""
        for _ in range(MAX_STRING_TOKENS):
            token = self.pick(
                self.vocab.quote_ids + self.vocab.plain_ids
            )
            if token in self.vocab.quote_ids:
                chunk = self.vocab.decode([token])
                index = _unescaped_quote_index(text, chunk)
                if index is not None:
                    self.log("stop string: closing quote won")
                    head = chunk[:index]
                    if head:
                        text += head
                        self.add(head)
                    break
            chunk = self._emit(token)
            if not text:
                chunk = chunk.lstrip(" ")
            text += chunk
        return _json_unescape(text)

    def gen_number(self, integer_only: bool) -> Any:
        """Generate a number; stop when the model picks an end token.

        Falls back to 0 if no digit was ever produced.
        """
        text = ""
        for _ in range(MAX_NUMBER_TOKENS):
            allowed = list(self.vocab.digit_ids)
            if not text and self.vocab.minus_id is not None:
                allowed.append(self.vocab.minus_id)
            if any(char.isdigit() for char in text):
                allowed += self.vocab.end_ids
                if (
                    not integer_only
                    and "." not in text
                    and self.vocab.dot_id is not None
                ):
                    allowed.append(self.vocab.dot_id)
            token = self.pick(allowed)
            if token in self.vocab.end_ids:
                break
            text += self._emit(token)
        if not any(char.isdigit() for char in text):
            return 0
        return int(text) if integer_only else float(text)

    def gen_value(self, schema: TypeSchema) -> Any:
        """Dispatch to the right constrained generator for
        `schema.type`."""
        if schema.type == "string":
            self.add('"')
            value = self.gen_string()
            self.add('"')
            return value
        if schema.type == "boolean":
            return self.choose(["true", "false"]) == "true"
        if schema.type == "null":
            return None
        return self.gen_number(schema.type == "integer")

    def log(self, message: str) -> None:
        """Print one generation step when --verbose is set."""
        if self.verbose:
            print(f"    -> {message}", file=sys.stderr)

    def run(self, prompt: str) -> JsonObject:
        """Generate one function call for the given prompt."""
        header = (
            "Pick the single function that best matches the "
            "request and answer with a JSON function call.\n"
            f"Available functions:\n{self.block}\n"
            f"Request: {prompt}\n"
            "Answer: "
        )
        self.ids = self.vocab.encode(header + '{"name": "')

        name = self.choose(list(self.functions))
        self.log(f"function = {name}")
        self.add('", "parameters": {')
        parameters: JsonObject = {}
        items = list(self.functions[name].parameters.items())
        for index, (key, schema) in enumerate(items):
            self.add(f"{json.dumps(key)}: ")
            parameters[key] = self.gen_value(schema)
            self.log(f"{key} = {parameters[key]!r}")
            if index < len(items) - 1:
                self.add(", ")
        self.add("}}")

        return {
            "prompt": prompt,
            "name": name,
            "parameters": parameters,
        }
