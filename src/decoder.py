"""Constrained token-by-token decoder for function-call JSON generation.

At every step the LLM produces logits over the full vocabulary. We set every
invalid token to -inf and take argmax — that is the entire masking idea.
"""

import json
from typing import Any, Dict, List, Set

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model

from src.parsing import FunctionDefinition
from src.trie import Trie, TrieNode, encode_ids

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

    Everything that can be precomputed (vocab sets, tries, functions block)
    is built in __init__ and reused for every prompt — no repeated work.
    """

    def __init__(
        self,
        llm: Small_LLM_Model,
        functions: Dict[str, FunctionDefinition],
    ) -> None:
        """Load vocab, build tries, precompute the functions description block."""
        self.llm = llm
        self.functions = functions
        self.ids: List[int] = []  # the growing token sequence, reset each run()

        # The vocabulary JSON maps token text → token id.
        # We need it to classify which ids are digits, dots, quotes, etc.
        try:
            with open(llm.get_path_to_vocab_file(), encoding="utf-8") as f:
                vocab: Dict[str, int] = json.load(f)
        except FileNotFoundError as e:
            raise OSError(f"Vocab file not found: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid vocab JSON: {e}") from e

        # Tokens whose entire text is digits: "0", "1", ..., "9", "42", ...
        # Multi-digit tokens like "42" are valid — they still only contain digits.
        self._digits: Set[int] = {
            i for t, i in vocab.items() if t and all(c.isdigit() for c in t)
        }

        # -1 means "not in this vocab" — safe because valid ids are >= 0.
        self._dot: int = vocab.get(".", -1)
        self._minus: int = vocab.get("-", -1)

        # End-of-number tokens: the first token id produced when encoding each
        # character. We only need the FIRST token because these are all single
        # ASCII characters, which tokenise to exactly one token each.
        self._end_ids: Set[int] = {
            encode_ids(llm, c)[0] for c in (",", "}", "]", " ", "\n")
        }

        # Any token whose text CONTAINS a double-quote can close a string.
        # BPE can bundle content + closing quote into one token (e.g. 'world"'),
        # so we match on containment, not equality.
        self._quote_ids: Set[int] = {i for t, i in vocab.items() if '"' in t}

        # Build the function-name trie once.  During generation, the model can
        # only pick token ids that keep at least one valid function name alive.
        self._fn_trie = Trie.from_strings(llm, list(functions))

        # Same idea for booleans — constrain to "true" or "false".
        self._bool_trie = Trie.from_strings(llm, ["true", "false"])

        # One-line description of each function, injected into every prompt so
        # the model has the context it needs to pick the right function.
        self._block = "\n".join(
            "- {}({}): {}".format(
                fn.name,
                ", ".join(f"{n}: {s.type}" for n, s in fn.parameters.items()),
                fn.description,
            )
            for fn in functions.values()
        )

    # ── core primitives ───────────────────────────────────────────────────────

    def _logits(self) -> Logits:
        """Ask the LLM for next-token logits given the current sequence."""
        return np.array(self.llm.get_logits_from_input_ids(self.ids), dtype=np.float64)

    def _pick(self, lg: Logits, allowed: Set[int]) -> int:
        """Constrained argmax: set every token outside `allowed` to -inf, return argmax.

        This IS constrained decoding.  The model still runs over the full
        vocabulary; we just ignore every token we don't want.
        """
        masked = np.full(len(lg), -np.inf)
        idx = list(allowed)
        masked[idx] = lg[idx]       # copy logits only for the valid tokens
        return int(np.argmax(masked))

    def _emit(self, text: str) -> None:
        """Append forced structural tokens to the sequence without calling the model.

        Positions where there is only one valid token (the opening '{', a key
        name, the separator ', ', etc.) don't need the model — we just write them.
        """
        self.ids += encode_ids(self.llm, text)

    def _walk(self, root: TrieNode) -> str:
        """Descend a trie one token at a time; call the model only at branch points.

        Forced step  (1 child)  — only one valid next token; skip the model.
        Branching step (>1 child) — real choice; run _pick over the children.

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
                tid = self._pick(self._logits(), set(node.children))
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
        for _ in range(_MAX_STRING):
            lg = self._logits()

            # Pick the best closing-quote token before we modify lg.
            close = self._pick(lg, self._quote_ids)
            close_val = float(lg[close])   # must save now — we mask in-place below

            # Set every quote token to -inf so argmax finds the best content token.
            lg[list(self._quote_ids)] = -np.inf
            best = int(np.argmax(lg))

            if close_val >= float(lg[best]):
                # Quote token won — the model is done with the string.
                # Salvage any content that appeared BEFORE the " in the closing token
                # (e.g. the "world" in 'world"') so the string value is complete.
                pre = self.llm.decode([close])
                q = pre.find('"')
                if q > 0:
                    text += pre[:q]
                    self._emit(pre[:q])
                break

            text += self.llm.decode([best])
            self.ids.append(best)
        return text

    def _number(self, *, integer_only: bool) -> float:
        """Decode a number; stop when the model picks an end token.

        The allowed set grows as the number is built:
          always          : digit tokens
          at start        : + minus (sign only valid at position 0)
          after a digit   : + end tokens (the model may stop here)
          after a digit,
          if not integer  : + dot (only once, never after the dot itself)

        The end token is NOT appended — the caller (_emit) writes the separator.
        """
        text, has_digit, has_dot = "", False, False
        for _ in range(_MAX_NUMBER):
            allowed: Set[int] = set(self._digits)
            if not text and self._minus != -1:
                allowed.add(self._minus)                 # sign only at start
            if not integer_only and has_digit and not has_dot and self._dot != -1:
                allowed.add(self._dot)                   # one dot, after a digit
            if has_digit:
                allowed |= self._end_ids                 # may stop after a digit

            tok = self._pick(self._logits(), allowed)
            if tok in self._end_ids:
                break                                    # end token → number done

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
        header = _INSTRUCTION.format(block=self._block, prompt=prompt)
        # Start the sequence with the full prompt + the opening of the JSON.
        # The model's next token must continue the function name.
        self.ids = encode_ids(self.llm, header + '{"name": "')

        # Let the model pick the function name, constrained to the trie.
        name = self._walk(self._fn_trie.root) or next(iter(self.functions))
        self._emit('", "parameters": {')   # structural tokens written by code

        fn = self.functions[name]
        items = list(fn.parameters.items())
        params: Dict[str, Any] = {}
        for idx, (key, schema) in enumerate(items):
            # json.dumps(key) adds the surrounding quotes and escapes if needed.
            self._emit(f"{json.dumps(key)}: ")
            if schema.type == "string":
                self._emit('"')                      # opening quote written by code
                params[key] = self._string()
                self._emit('"')                      # closing quote written by code
            elif schema.type == "boolean":
                params[key] = self._walk(self._bool_trie.root) == "true"
            else:
                params[key] = self._number(integer_only=schema.type == "integer")
            if idx < len(items) - 1:
                self._emit(", ")                     # separator between params
        self._emit("}}")                             # close parameters + root object

        return {"prompt": prompt, "name": name, "parameters": params}
