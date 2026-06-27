import json
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.parsing import FunctionDefinition
from src.trinode import Trie


JsonObject = Dict[str, Any]


class GenerationContext(BaseModel):
    """Reusable constraints and prompt text shared across all prompts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: Dict[str, FunctionDefinition]
    function_trie: Trie
    boolean_trie: Trie
    functions_block: str


def token_ids(llm: Small_LLM_Model, text: str) -> List[int]:
    """Encode text and return a plain list of token ids."""
    encoded = llm.encode(text)
    raw_ids = encoded.tolist()
    if raw_ids and isinstance(raw_ids[0], list):
        return [int(token_id) for token_id in raw_ids[0]]
    return [int(token_id) for token_id in raw_ids]


def mask_logits(logits: List[float], allowed_ids: List[int]) -> int:
    """Return the best model token from the currently allowed ids."""
    if not allowed_ids:
        raise ValueError("Error: constrained decoder has no valid next token")
    if len(allowed_ids) < 1000:
        best_id = allowed_ids[0]
        best_val = logits[best_id]
        for token_id in allowed_ids[1:]:
            if logits[token_id] > best_val:
                best_val = logits[token_id]
                best_id = token_id
        return best_id
    arr = np.array(logits, dtype=np.float32)
    return int(allowed_ids[int(np.argmax(arr[allowed_ids]))])


def pick_from_trie(
    llm: Small_LLM_Model, trie: Trie, input_ids: List[int]
) -> str:
    """Generate tokens constrained by a trie until a leaf is reached."""
    node = trie.root
    generated: List[int] = []
    while (allowed := trie.allowed_next_ids(node)):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = mask_logits(logits, allowed)
        node = trie.step(node, next_id)
        generated.append(next_id)
    if node.value is None:
        raise ValueError("Trie generation ended without reaching a leaf")
    return node.value


def build_trie(
    llm: Small_LLM_Model, prefix: str, entries: Dict[str, str]
) -> Trie:
    """Build a trie from a label-to-value mapping with a shared prefix."""
    trie = Trie()
    prefix_ids = token_ids(llm, prefix)
    for label, value in entries.items():
        full_ids = token_ids(llm, prefix + label)
        trie.insert(full_ids[len(prefix_ids):], value)
    return trie


def build_generation_context(
    llm: Small_LLM_Model,
    functions: Dict[str, FunctionDefinition],
) -> GenerationContext:
    """Build constraints reused for every prompt in the input file."""
    function_trie = build_trie(
        llm, '\nJSON: { "name": "', {name: name for name in functions}
    )
    boolean_trie = build_trie(llm, '": ', {"true": "true", "false": "false"})
    function_list = [
        function.model_dump() for function in functions.values()
    ]
    return GenerationContext(
        functions=functions,
        function_trie=function_trie,
        boolean_trie=boolean_trie,
        functions_block=json.dumps(function_list, separators=(",", ":")),
    )
