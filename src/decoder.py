"""Constrained token-by-token decoder for function-call JSON generation.

The compile phase (src/grammar.py) has already turned the schema into
vocabulary masks, trie branch masks, and pre-encoded literal spans. This
file only walks that compiled Grammar: literal spans are appended for free
(no model call), and at every genuine choice — which name, which digits,
which string chars, true vs false — we ask the model for logits once and
pick from a mask that already exists.
"""

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model

from src.grammar import Grammar, Mask, compile_grammar
from src.parsing import FunctionDefinition
from src.trie import TrieNode, encode_ids

JsonObject = Dict[str, Any]
Logits = npt.NDArray[np.float64]

# The prompt template that gives the model context before the JSON we build.
# {block} = one-line description of each function.
# {prompt} = the user's natural-language request.
_INSTRUCTION = (
    "You convert the request into a function call.\n"
    "Available functions:\n{block}\n"
    "Request: {prompt}\n"
)

# Hard caps so a misbehaving model never loops forever.
_MAX_STRING = 64
_MAX_NUMBER = 24


class Decoder:
    """Set up once per run; call run(prompt) for each prompt.

    The compile phase (Grammar) builds every mask and literal span up
    front, in __init__. run() never repeats that work — it only walks the
    compiled structure.
    """

    def __init__(
        self,
        llm: Small_LLM_Model,
        functions: Dict[str, FunctionDefinition],
    ) -> None:
        """Probe the model's real vocab width, then compile the grammar."""
        self.llm = llm
        self.functions = functions
        self.ids: List[int] = []  # the growing token sequence, reset each run()

        # The tokenizer's vocab file can under-count the model's actual
        # logits width (added special tokens usually aren't listed in it),
        # so masks are sized from one real forward pass instead of a guess.
        # This is the only model call in __init__ — it happens once per run,
        # never once per prompt.
        probe_ids = encode_ids(llm, "0")
        vocab_size = len(llm.get_logits_from_input_ids(probe_ids))

        self.grammar: Grammar = compile_grammar(llm, functions, vocab_size)

    # ── core primitives ───────────────────────────────────────────────────────

    def _logits(self) -> Logits:
        """Ask the LLM for next-token logits given the current sequence."""
        return np.array(self.llm.get_logits_from_input_ids(self.ids), dtype=np.float64)

    def _pick(self, lg: Logits, mask: Mask) -> int:
        """Constrained argmax over a precomputed boolean mask.

        This IS constrained decoding.  The model still runs over the full
        vocabulary; the mask (built once at compile time) just tells argmax
        which tokens to ignore.
        """
        return int(np.argmax(np.where(mask, lg, -np.inf)))

    def _emit(self, ids: List[int]) -> None:
        """Append a pre-encoded literal span without calling the model.

        Positions where there is only one valid token (the opening '{', a
        key name, the separator ', ', etc.) were already encoded once by
        the compile phase — we just extend the sequence.
        """
        self.ids += ids

    def _walk(self, root: TrieNode) -> str:
        """Descend a trie one token at a time; call the model only at branch points.

        Forced step  (1 child)  — only one valid next token; skip the model.
        Branching step (>1 child) — real choice; pick from the cached mask.

        This handles multi-token words (e.g. "fn_add_numbers" is several tokens)
        while keeping model calls to the minimum needed.
        """
        node = root
        while node.children:
            if len(node.children) == 1:
                # Only one valid continuation — no need to ask the model.
                (tid, child), = node.children.items()
            else:
                # Real choice: the model's logits decide which branch to take.
                assert node.mask is not None  # set by the compile phase
                tid = self._pick(self._logits(), node.mask)
                child = node.children[tid]
            self.ids.append(tid)
            node = child
        return node.value or ""  # leaf stores the complete word

    # ── value decoders ────────────────────────────────────────────────────────

    def _string(self) -> str:
        """Decode a string value token by token; stop when a closing quote wins.

        At each step we get one logit vector and find two candidates:
          close — best token among those that CONTAIN a quote (potential closers)
          best  — best token among ALL others (content)

        If close beats best, the model wants to end the string → stop.
        Otherwise append best and continue.

        BPE tokens can bundle content + closing quote (e.g. 'world"').  When
        that wins, we salvage the text before the quote so nothing is lost.
        """
        text = ""
        g = self.grammar
        for _ in range(_MAX_STRING):
            lg = self._logits()

            close = self._pick(lg, g.quote_mask)
            close_val = float(lg[close])
            best = self._pick(lg, g.content_mask)

            if close_val >= float(lg[best]):
                # Quote token won — the model is done with the string.
                # Salvage any content that appeared BEFORE the " in the closing token
                # (e.g. the "world" in 'world"') so the string value is complete.
                pre = self.llm.decode([close])
                q = pre.find('"')
                if q > 0:
                    text += pre[:q]
                    self._emit(encode_ids(self.llm, pre[:q]))
                break

            text += self.llm.decode([best])
            self.ids.append(best)
        return text

    def _number(self, *, integer_only: bool) -> float:
        """Decode a number; stop when the model picks an end token.

        The FSM has exactly four states, each with a mask built once at
        compile time:
          start           : digits + minus (sign only valid at position 0)
          after '-' only  : digits only (no second sign, nothing to stop yet)
          after a digit,
          float, no dot   : digits + dot + end tokens
          after a digit,
          dot used / int  : digits + end tokens (no more dots ever)

        The end token is NOT appended — the caller (_emit) writes the separator.
        """
        text, has_digit, has_dot = "", False, False
        g = self.grammar
        for _ in range(_MAX_NUMBER):
            if not text:
                mask = g.num_start_mask
            elif not has_digit:
                mask = g.digit_mask
            elif not integer_only and not has_dot:
                mask = g.num_pre_dot_mask
            else:
                mask = g.num_post_mask

            tok = self._pick(self._logits(), mask)
            if has_digit and g.end_mask[tok]:
                break  # end token → number done

            piece = self.llm.decode([tok])
            text += piece
            self.ids.append(tok)
            has_digit = has_digit or any(c.isdigit() for c in piece)
            has_dot = has_dot or "." in piece

        # If no digit was ever seen (edge case), default to 0.
        return (int(text) if integer_only else float(text)) if text else 0.0

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, prompt: str) -> JsonObject:
        """Generate one function-call JSON object for the given prompt.

        The JSON skeleton is written by the code; the model only fills in
        the content positions (function name, parameter values) under constraint.

        Output shape:
          {"name": "<fn>", "parameters": {"key": <value>, ...}}
        """
        g = self.grammar
        header = _INSTRUCTION.format(block=g.block, prompt=prompt)
        # Start the sequence with the full prompt + the opening of the JSON.
        # The model's next token must continue the function name.
        self.ids = encode_ids(self.llm, header + '{"name": "')

        # Let the model pick the function name, constrained to the trie.
        name = self._walk(g.fn_trie.root) or next(iter(self.functions))
        self._emit(g.head_ids)  # structural tokens, pre-encoded at compile time

        fn = self.functions[name]
        items = list(fn.parameters.items())
        prefixes = g.param_prefix[name]
        params: Dict[str, Any] = {}
        for idx, (key, schema) in enumerate(items):
            self._emit(prefixes[idx])             # '"key": ', pre-encoded
            if schema.type == "string":
                self._emit(g.quote_ids)                  # opening quote
                params[key] = self._string()
                self._emit(g.quote_ids)                  # closing quote
            elif schema.type == "boolean":
                params[key] = self._walk(g.bool_trie.root) == "true"
            else:
                params[key] = self._number(integer_only=schema.type == "integer")
            if idx < len(items) - 1:
                self._emit(g.sep_ids)              # separator between params
        self._emit(g.tail_ids)                     # close parameters + root object

        return {"prompt": prompt, "name": name, "parameters": params}
