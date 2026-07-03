from _typeshed import SupportsWrite
from os import path
from typing import ItemsView

from llm_sdk import Small_LLM_Model
from parsing import FunctionDefinition, Prompt
import numpy as np
import json


class Vocab:
    def __init__(self, llm: Small_LLM_Model) -> None:
        self.llm = llm
        self.vocab_file = self.llm.get_path_to_vocab_file()
        with open(self.vocab_file) as f:
            self.vocab: dict[str, int] = json.load(f)
        self.digits: list[int] = [
            i for token, i in self.vocab.items() if token.isdigit()
        ]
        self.quotes: list[int] = [i for token, i in self.vocab.items() if '"' in token]

        self.plain_ids: list[int] = [
            i for token, i in self.vocab.items() if '"' not in token
        ]
        self.dot_id = self.vocab.get(".")
        self.minus_id = self.vocab.get("-")
        self.end = [",", "}"]
        self.end_ids = [i for token, i in self.vocab.items() if token in self.end]

    def pick(self, ids, allowed):
        logits = np.array(self.llm.get_logits_from_input_ids(ids))
        masked = np.full(len(logits), -np.inf)
        masked[allowed] = logits[allowed]
        return int(np.argmax(masked))

    def gen_number(self, integer_only, ids, max_steps=24):
        text = ""
        for _ in range(max_steps):
            allowed = list(self.digits)
            if text == "":
                if self.minus_id is not None:
                    allowed = self.digits + [self.minus_id]
            if any(c.isdigit() for c in text):
                allowed += self.end_ids
                if not integer_only and "." not in text and self.dot_id is not None:
                    allowed.append(self.dot_id)
            token = self.pick(ids, allowed)
            if token in self.end_ids:
                break
            piece = self.llm.decode([token])
            text += piece
            ids.append(token)
        if text == "":
            raise ValueError("Fuck u")
        return int(text) if integer_only else float(text)

    def gen_string(self, ids, max_steps=64) -> str:
        text = ""
        for _ in range(max_steps):
            logits = np.array(self.llm.get_logits_from_input_ids(ids))
            best_one = self.plain_ids[int(np.argmax(logits[self.plain_ids]))]
            best_closed = self.quotes[int(np.argmax(logits[self.quotes]))]
            if logits[best_closed] >= logits[best_one]:
                token = best_closed
                head = self.llm.decode([token]).split('"')[0]
                if head:
                    text += head
                break
            else:
                token = best_one
                piece = self.llm.decode([token])
                text += piece
                ids.append(token)
        if text == "":
            raise ValueError("Error")
        return text

    def chose(self, ids, options: list[str]) -> str:
        paths = {option: self.llm.encode(option) for option in options}
        step = 0
        while len(paths) > 1:
            allowed = sorted({ids[step] for ids in paths.values()})
            token = allowed[0] if len(allowed) == 0 else self.pick(allowed)
            self.ids.append(token)
            paths = {o: ids for o, ids in paths.items() if ids[step] == token}


llm = Small_LLM_Model()
v = Vocab(llm)
prompt = (
    "You convert the request into a function call.\n"
    "Available functions:\n- get_weather(city: string): get weather for a city\n"
    "Request: what's the weather in Rabat?\n"
)
ids = llm.encode(prompt + '{"name": "get_weather", "parameters": {"city": "').tolist()[
    0
]
result = v.gen_string(ids)
print(repr(result))
