"""Constrained token-by-token decoding: given a prompt and a catalog
of callable functions, generate a JSON function call one token at a
time, masking the model's logits so only grammatically valid tokens
can ever be picked."""

import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.parsing import FunctionDefinition, TypeSchema
from src.trie import Trie
from src.vocab import Vocab

JsonObject = Dict[str, Any]

MAX_STRING_TOKENS = 64
MAX_NUMBER_TOKENS = 24


def _trailing_backslashes(text: str) -> int:
    """Count the consecutive backslashes at the end of a string.

    Args:
        text: The string to inspect.

    Returns:
        How many backslashes end the string (0 if none).
    """
    return len(text) - len(text.rstrip("\\"))


def _unescaped_quote_index(text: str, chunk: str) -> Optional[int]:
    """Find the first quote in `chunk` that is not escaped.

    Backslashes already emitted at the end of `text` count toward
    the escaping of the first characters of `chunk`.

    Args:
        text: The string content generated so far.
        chunk: The decoded text of the candidate token.

    Returns:
        The index in `chunk` of the first unescaped quote, or None
        if every quote in `chunk` is escaped (or there is none).
    """
    backslashes = _trailing_backslashes(text)
    for index, char in enumerate(chunk):
        if char == '"' and backslashes % 2 == 0:
            return index
        backslashes = backslashes + 1 if char == "\\" else 0
    return None


def _json_unescape(text: str) -> str:
    """Decode JSON string escapes (e.g. ``\\\\`` -> ``\\``).

    Args:
        text: The raw string content, as emitted by the model.

    Returns:
        The unescaped string, or `text` unchanged if the model
        produced escapes JSON does not accept.
    """
    try:
        return str(json.loads(f'"{text}"'))
    except ValueError:
        return text


class Decoder(BaseModel):
    """Create once per run, then call run(prompt) for each prompt.

    Attributes:
        functions: The callable function catalog, keyed by name.
        vocab: The tokenizer/vocab wrapper used to mask logits.
        verbose: Whether to print each generation step to stderr.
        ids: Token ids of the current generation; reset by run().
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: Dict[str, FunctionDefinition]
    vocab: Vocab
    verbose: bool = False
    ids: List[int] = Field(default_factory=list)

    @property
    def block(self) -> str:
        """Render the function catalog as prompt text.

        Returns:
            One line per function: name, typed parameters, and
            description.
        """
        lines = []
        for fn in self.functions.values():
            params = ", ".join(
                f"{name}: {schema.type}"
                for name, schema in fn.parameters.items()
            )
            lines.append(f"- {fn.name}({params}): {fn.description}")
        return "\n".join(lines)

    def log(self, message: str) -> None:
        """Print one generation step to stderr when verbose is set.

        Args:
            message: The step description to print.

        Returns:
            None.
        """
        if self.verbose:
            print(f"    -> {message}", file=sys.stderr)

    def add(self, text: str) -> None:
        """Append fixed JSON structure without consulting the model.

        Args:
            text: The literal text to encode and append.

        Returns:
            None.
        """
        self.ids += self.vocab.encode(text)

    def pick(self, allowed: List[int]) -> int:
        """Run one constrained decoding step.

        Fetches the next-token logits for the current context, masks
        every token outside `allowed` to -inf, and takes the argmax.

        Args:
            allowed: The token ids the grammar permits here.

        Returns:
            The id of the highest-logit token among `allowed`.

        Raises:
            ValueError: If `allowed` is empty (argmax over an all
                -inf array would silently return token 0).
        """
        if not allowed:
            raise ValueError("Error: pick() got no allowed tokens")
        logits = np.array(
            self.vocab.llm.get_logits_from_input_ids(self.ids)
        )
        masked = np.full(len(logits), -np.inf)
        masked[allowed] = logits[allowed]
        token = int(np.argmax(masked))
        self.log(
            f"pick: {len(allowed)} allowed token(s) -> chose id {token} "
            f"({self.vocab.decode([token])!r})"
        )
        return token

    def _emit(self, token: int) -> str:
        """Accept one model-chosen token into the context.

        Args:
            token: The token id to append.

        Returns:
            The decoded text of the appended token.
        """
        self.ids.append(token)
        text = self.vocab.decode([token])
        self.log(f"emit: id {token} decodes to {text!r}")
        return text

    def choose(self, options: List[str]) -> str:
        """Let the model pick one of `options` (e.g. a function name).

        Every option is encoded into a token trie; at each step only
        the tokens that continue at least one still-possible option
        are allowed. Where one option is a completed prefix of
        another, a closing-quote token is also allowed and means
        "stop here".

        Args:
            options: The candidate strings; must not be empty.

        Returns:
            The option the model selected.

        Raises:
            ValueError: If `options` is empty.
        """
        if not options:
            raise ValueError(
                "Error: choose() needs at least one option"
            )

        trie = Trie()
        for option in options:
            trie.insert(self.vocab.encode(option), option)

        node = trie.root
        while True:
            if node.value is not None and not node.children:
                return node.value
            allowed = sorted(node.children)
            if node.value is not None:
                allowed += self.vocab.quote_ids
            token = (
                allowed[0]
                if len(allowed) == 1
                else self.pick(allowed)
            )
            if node.value is not None and token not in node.children:
                return node.value
            self.ids.append(token)
            node = node.children[token]

    def gen_string(self) -> str:
        """Generate one JSON string value (without its quotes).

        pick() runs over the full vocabulary; generation stops on
        the first unescaped quote, then JSON escapes are decoded.
        Bounded by MAX_STRING_TOKENS so a stubborn model cannot
        hang.

        Returns:
            The unescaped string content chosen by the model.
        """
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
        else:
            if _trailing_backslashes(text) % 2:
                self.add("\\")
        return _json_unescape(text)

    def gen_number(self, integer_only: bool) -> int | float:
        """Generate one JSON number value.

        Digits stay allowed throughout, a minus is allowed only
        first, and a dot only once and only for floats. End tokens
        become allowed once a digit exists; picking one stops
        generation. Bounded by MAX_NUMBER_TOKENS so a stubborn model
        cannot hang.

        Args:
            integer_only: If True, forbid the decimal dot and parse
                the result as an int; otherwise parse it as a float.

        Returns:
            The parsed number, or 0 if no digit was ever produced.
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
        """Generate one value of the type declared by `schema`.

        Strings are wrapped in quotes, booleans go through choose(),
        null is appended as a literal with no model call, and both
        number types go through gen_number().

        Args:
            schema: The declared type of the value to generate.

        Returns:
            The generated Python value: str, bool, None, int, or
            float.
        """
        if schema.type == "string":
            self.add('"')
            value = self.gen_string()
            self.add('"')
            return value
        if schema.type == "boolean":
            return self.choose(["true", "false"]) == "true"
        if schema.type == "null":
            self.add("null")
            return None
        return self.gen_number(schema.type == "integer")

    def run(self, prompt: str) -> JsonObject:
        """Generate one function call for `prompt`.

        Builds the instruction header, lets the model choose a
        function name, then generates each declared parameter in
        order inside a fixed JSON scaffold.

        Args:
            prompt: The natural-language request.

        Returns:
            A dict with the keys "prompt", "name", and "parameters".
        """
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
        schemas = self.functions[name].parameters
        for index, (key, schema) in enumerate(schemas.items()):
            if index:
                self.add(", ")
            self.add(f"{json.dumps(key)}: ")
            parameters[key] = self.gen_value(schema)
            self.log(f"{key} = {parameters[key]!r}")
        self.add("}}")

        return {
            "prompt": prompt,
            "name": name,
            "parameters": parameters,
        }
