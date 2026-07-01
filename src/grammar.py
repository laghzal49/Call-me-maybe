"""Compile phase: turn the function schema into vocabulary-wide constraints.

Every decode step lands in one of a handful of FSM states — start of a
number, after a digit, inside a string, at a trie branch — and those states
repeat constantly: across every value, across every prompt.  Recomputing a
Python set (or a fresh -inf array) for the same state on every step wastes
work that never changes.

This module builds each state's constraint exactly once, as a boolean mask
the width of the vocabulary, so the decode loop (src/decoder.py) never does
more than index into an array that already exists.  It also pre-encodes the
literal spans of the output template (`", "parameters": {`, `"key": `, the
separators, the closing braces) — text with zero entropy that the decoder
appends without ever calling the model.
"""

import json
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model

from src.parsing import FunctionDefinition
from src.trie import Trie, TrieNode, encode_ids

Mask = npt.NDArray[np.bool_]


class Grammar:
    """Prompt-independent constraints, built once and shared by every run().

    Everything here is the same for every prompt in the run: it only
    depends on the model's vocabulary and the function definitions.
    """

    def __init__(
        self,
        llm: Small_LLM_Model,
        functions: Dict[str, FunctionDefinition],
        vocab_size: int,
    ) -> None:
        try:
            with open(llm.get_path_to_vocab_file(), encoding="utf-8") as f:
                vocab: Dict[str, int] = json.load(f)
        except FileNotFoundError as e:
            raise OSError(f"Vocab file not found: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid vocab JSON: {e}") from e

        self.vocab_size = vocab_size

        # ── vocabulary-wide masks, one per distinct decode state ───────────
        digit_ids = [i for t, i in vocab.items() if t and all(c.isdigit() for c in t)]
        quote_ids = [i for t, i in vocab.items() if '"' in t]
        end_ids = [encode_ids(llm, c)[0] for c in (",", "}", "]", " ", "\n")]
        dot_id = vocab.get(".", -1)
        minus_id = vocab.get("-", -1)

        self.digit_mask = self._mask(digit_ids)
        self.quote_mask = self._mask(quote_ids)
        self.content_mask = ~self.quote_mask
        self.end_mask = self._mask(end_ids)

        # Number FSM: no matter how many digits have been produced so far,
        # only three token sets ever matter — precompute all three.
        self.num_start_mask = self.digit_mask.copy()          # position 0
        if minus_id != -1:
            self.num_start_mask[minus_id] = True

        self.num_post_mask = self.digit_mask | self.end_mask  # after digit,
        #                                                        dot used up or integer

        self.num_pre_dot_mask = self.num_post_mask.copy()     # after digit,
        if dot_id != -1:                                      # dot still legal (float)
            self.num_pre_dot_mask[dot_id] = True

        # ── fixed-choice tries (function names, booleans) ──────────────────
        # Branching nodes get a cached mask so _pick never rebuilds one from
        # node.children; forced (single-child) nodes stay free — no model
        # call, no mask needed.
        self.fn_trie = Trie.from_strings(llm, list(functions))
        self.bool_trie = Trie.from_strings(llm, ["true", "false"])
        self._annotate(self.fn_trie)
        self._annotate(self.bool_trie)

        # ── prompt-block text (same for every prompt) ───────────────────────
        self.block = "\n".join(
            "- {}({}): {}".format(
                fn.name,
                ", ".join(f"{n}: {s.type}" for n, s in fn.parameters.items()),
                fn.description,
            )
            for fn in functions.values()
        )

        # ── literal output-template spans, pre-encoded once ─────────────────
        # These positions have zero entropy: once the function is picked,
        # everything below is forced JSON structure written by the code.
        self.head_ids = encode_ids(llm, '", "parameters": {')
        self.sep_ids = encode_ids(llm, ", ")
        self.tail_ids = encode_ids(llm, "}}")
        self.quote_ids = encode_ids(llm, '"')
        self.param_prefix: Dict[str, List[List[int]]] = {
            fn.name: [encode_ids(llm, f"{json.dumps(key)}: ") for key in fn.parameters]
            for fn in functions.values()
        }

    def _mask(self, ids: List[int]) -> Mask:
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[ids] = True
        return mask

    def _annotate(self, trie: Trie) -> None:
        """Walk a trie once, caching a boolean mask on every branching node."""
        stack: List[TrieNode] = [trie.root]
        while stack:
            node = stack.pop()
            if len(node.children) > 1:
                node.mask = self._mask(list(node.children))
            stack.extend(node.children.values())


def compile_grammar(
    llm: Small_LLM_Model,
    functions: Dict[str, FunctionDefinition],
    vocab_size: int,
) -> Grammar:
    """Compile once per run, before any prompt is processed."""
    return Grammar(llm, functions, vocab_size)
