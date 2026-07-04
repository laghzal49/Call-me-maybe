import json
from typing import Any, Dict, List

import numpy as np

from llm_sdk import Small_LLM_Model
from src.parsing import FunctionDefinition

JsonObject = Dict[str, Any]

MAX_STRING_TOKENS = 64
MAX_NUMBER_TOKENS = 24


class Decoder:
    """Create once per run, then call run(prompt) for each prompt."""

    def __init__(
        self,
        llm: Small_LLM_Model,
        functions: Dict[str, FunctionDefinition],
    ) -> None:
        """Load the vocabulary and group the token ids we will allow."""
        self.llm = llm
        self.functions = functions
        self.ids: List[int] = []

        try:
            with open(llm.get_path_to_vocab_file(), encoding="utf-8") as file:
                vocab: Dict[str, int] = json.load(file)
        except OSError as error:
            raise ValueError(
                f"Error: cannot read vocab file: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Error: invalid vocab JSON: {error}") from error
        self.digit_ids = [i for tok, i in vocab.items() if tok.isdigit()]
        self.quote_ids = [i for tok, i in vocab.items() if '"' in tok]
        self.plain_ids = [i for tok, i in vocab.items() if '"' not in tok]
        self.dot_id = vocab.get(".")
        self.minus_id = vocab.get("-")
        self.end_ids = [self.encode(char)[0] for char in (",", "}", " ", "\n")]
        self.block = "\n".join(
            "- {}({}): {}".format(
                fn.name,
                ", ".join(f"{k}: {v.type}" for k, v in fn.parameters.items()),
                fn.description,
            )
            for fn in self.functions.values()
        )

    def encode(self, text: str) -> List[int]:
        """Encode text to a flat list of token ids."""
        ids: List[int] = self.llm.encode(text).tolist()[0]
        return ids

    def add(self, text: str) -> None:
        """Append fixed JSON structure without asking the model."""
        self.ids += self.encode(text)

    def pick(self, allowed: List[int]) -> int:
        """One constrained decoding step."""
        logits = np.array(self.llm.get_logits_from_input_ids(self.ids))
        masked = np.full(len(logits), -np.inf)
        masked[allowed] = logits[allowed]
        return int(np.argmax(masked))

    def choose(self, options: List[str]) -> str:
        """Let the model pick one of `options` (e.g. a function name)."""
        paths = {option: self.encode(option) for option in options}
        step = 0
        while len(paths) > 1:
            allowed = sorted({ids[step] for ids in paths.values()})
            token = allowed[0] if len(allowed) == 1 else self.pick(allowed)
            self.ids.append(token)
            paths = {o: ids for o, ids in paths.items() if ids[step] == token}
            step += 1
        choice, ids = paths.popitem()
        self.ids += ids[step:]
        return choice

    def gen_string(self) -> str:
        """Generate a string value; stop when the model wants to close it."""
        text = ""
        for _ in range(MAX_STRING_TOKENS):
            logits = np.array(self.llm.get_logits_from_input_ids(self.ids))
            closing = self.quote_ids[int(np.argmax(logits[self.quote_ids]))]
            content = self.plain_ids[int(np.argmax(logits[self.plain_ids]))]
            if logits[closing] >= logits[content]:
                head = self.llm.decode([closing]).split('"')[0]
                if head:
                    text += head
                    self.add(head)
                break
            text += self.llm.decode([content])
            self.ids.append(content)
        return text

    def gen_number(self, integer_only: bool) -> Any:
        """Generate a number; stop when the model picks an end token."""
        text = ""
        for _ in range(MAX_NUMBER_TOKENS):
            allowed = list(self.digit_ids)
            if not text and self.minus_id is not None:
                allowed.append(self.minus_id)
            if any(char.isdigit() for char in text):
                allowed += self.end_ids
                if not integer_only and "." not \
                        in text and self.dot_id is not None:
                    allowed.append(self.dot_id)
            token = self.pick(allowed)
            if token in self.end_ids:
                break
            text += self.llm.decode([token])
            self.ids.append(token)
        if not any(char.isdigit() for char in text):
            return 0
        return int(text) if integer_only else float(text)

    def run(self, prompt: str) -> JsonObject:
        """Generate one function call for the given prompt."""
        header = (
            "You convert the request into a function call.\n"
            f"Available functions:\n{self.block}\n"
            f"Request: {prompt}\n"
        )
        self.ids = self.encode(header + '{"name": "')

        name = self.choose(list(self.functions))
        self.add('", "parameters": {')

        parameters: JsonObject = {}
        items = list(self.functions[name].parameters.items())
        for index, (key, schema) in enumerate(items):
            self.add(f"{json.dumps(key)}: ")
            if schema.type == "string":
                self.add('"')
                parameters[key] = self.gen_string()
                self.add('"')
            elif schema.type == "boolean":
                parameters[key] = self.choose(["true", "false"]) == "true"
            else:
                parameters[key] = self.gen_number(schema.type == "integer")
            if index < len(items) - 1:
                self.add(", ")
        self.add("}}")

        return {"prompt": prompt, "name": name, "parameters": parameters}
