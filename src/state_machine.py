"""The state machine that builds one function call token by token.

We keep ONE growing list of token ids (`self.ids`) and only append to it. The
fixed JSON structure (braces, quotes, keys) is written directly; the model is
asked only where it has a real choice (the name and the values), and there we
mask the logits so only valid tokens can be picked.
"""

import json
from enum import Enum, auto
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model

from src.context import INSTRUCTION, GenerationContext
from src.masking import pick_allowed, pick_excluding
from src.parsing import TypeSchema
from src.trie import TrieNode, encode_ids
from src.vocab import Vocab

JsonObject = Dict[str, Any]

# hard caps so a misbehaving model can never loop forever
MAX_NUMBER_TOKENS = 24
MAX_STRING_TOKENS = 64


class State(Enum):
    """The phases the machine moves through."""

    NAME = auto()      # choosing the function name
    PARAMS = auto()    # filling each parameter value
    DONE = auto()      # finished


class StateMachine:
    """Builds one function call by extending a single token sequence."""

    def __init__(
        self,
        llm: Small_LLM_Model,
        vocab: Vocab,
        ctx: GenerationContext,
        prompt: str,
    ) -> None:
        """Start the sequence at the prompt header + the JSON opening."""
        self.llm = llm
        self.vocab = vocab
        self.ctx = ctx
        self.state = State.NAME
        # the call we will return at the end
        self.result: JsonObject = {"prompt": prompt, "name": "", "parameters": {}}
        # the single, append-only token sequence we keep feeding the model
        header = INSTRUCTION.format(block=ctx.functions_block, prompt=prompt)
        self.ids: List[int] = encode_ids(llm, header + '{"name": "')

    # --- low-level helpers -------------------------------------------------

    def emit(self, text: str) -> None:
        """Append fixed structural text (we are not asking the model here)."""
        self.ids += encode_ids(self.llm, text)

    def logits(self) -> List[float]:
        """Next-token logits for the current sequence."""
        return self.llm.get_logits_from_input_ids(self.ids)

    # --- value decoders (each appends only the chosen value tokens) --------

    def walk_trie(self, root: TrieNode) -> str:
        """Descend a trie; ask the model only where there is a real choice."""
        node = root
        while node.children:
            if len(node.children) == 1:
                # only one valid token -> forced, no need to run the model
                (token_id, child), = node.children.items()
            else:
                # real choice: run the model and mask to the valid children
                token_id = pick_allowed(self.logits(), set(node.children))
                child = node.children[token_id]
            self.ids.append(token_id)   # continue the same sequence
            node = child
        return node.value or ""         # the word stored at the leaf

    def decode_string(self) -> str:
        """Emit content tokens; stop when the model prefers to close the string.

        A string ends with a quote token (e.g. '"', '",', '"}'). We compare the
        best plain-content token (no quote) against the best closing token (has a
        quote): if closing wins, the value is finished. The closing quote itself
        is written as structure by `do_params`, not here.
        """
        text = ""
        for _ in range(MAX_STRING_TOKENS):
            logits = self.logits()
            best_content = pick_excluding(logits, self.vocab.string_quote_ids)
            best_close = pick_allowed(logits, self.vocab.string_quote_ids)
            if logits[best_close] >= logits[best_content]:
                break   # model wants to end the string
            text += self.vocab.decode_token(best_content)
            self.ids.append(best_content)
        return text

    def decode_number(self, *, integer_only: bool) -> float:
        """Allow digits/dot/sign; stop when the model picks an end token."""
        text = ""
        has_digit = has_dot = False
        for _ in range(MAX_NUMBER_TOKENS):
            started = bool(text)   # has anything been emitted yet?
            if integer_only:
                allowed = self.vocab.integer_tokens(started=started)
            else:
                allowed = self.vocab.number_tokens(
                    started=started, has_digit=has_digit, has_dot=has_dot
                )
            if has_digit:          # after a digit, the model may stop
                allowed = allowed | self.vocab.number_end_ids
            pick = pick_allowed(self.logits(), allowed)
            if pick in self.vocab.number_end_ids:   # end token -> number done
                break
            piece = self.vocab.decode_token(pick)
            text += piece
            self.ids.append(pick)
            if any(c.isdigit() for c in piece):
                has_digit = True
            if "." in piece:
                has_dot = True
        if not has_digit:          # no digits seen -> default 0
            return 0
        return int(text) if integer_only else float(text)

    def decode_value(self, schema: TypeSchema) -> Any:
        """Pick the decoder matching the parameter's declared type."""
        if schema.type == "string":
            return self.decode_string()
        if schema.type == "boolean":
            return self.walk_trie(self.ctx.boolean_trie.root) == "true"
        return self.decode_number(integer_only=schema.type == "integer")

    # --- phase handlers ----------------------------------------------------

    def do_name(self) -> None:
        """Choose the function name, then open the parameters object."""
        name = self.walk_trie(self.ctx.function_trie.root)
        # fall back to the first function if the walk found nothing
        self.result["name"] = name or next(iter(self.ctx.functions))
        self.emit('", "parameters": {')

    def do_params(self) -> None:
        """Fill every parameter value in order, then close the JSON."""
        function = self.ctx.functions[self.result["name"]]
        items = list(function.parameters.items())
        for index, (key, schema) in enumerate(items):
            self.emit(f"{json.dumps(key)}: ")    # e.g.  "a":
            is_string = schema.type == "string"
            if is_string:
                self.emit('"')                    # open the string ourselves
            value = self.decode_value(schema)
            if is_string:
                self.emit('"')                    # close it ourselves
            self.result["parameters"][key] = value
            if index < len(items) - 1:            # comma before the next value
                self.emit(", ")
        self.emit("}}")                           # close parameters + root

    # --- driver ------------------------------------------------------------

    def run(self) -> JsonObject:
        """Step through the phases until done, then return the call."""
        while self.state != State.DONE:
            if self.state == State.NAME:
                self.do_name()
                self.state = State.PARAMS
            else:                                 # State.PARAMS
                self.do_params()
                self.state = State.DONE
        return self.result
