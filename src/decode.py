from typing import List, Dict, Any
import numpy as np
import json
import re

from llm_sdk import Small_LLM_Model
from src.trinode import Trie
from src.vocab import Vocab
from src.parsing import FunctionDefinition


def _mask_logits(logits: List[float], allowed_ids: List[int]) -> int:
    """Return argmax token id among allowed_ids."""
    if not allowed_ids:
        return 0
    if len(allowed_ids) < 1000:
        best_id = allowed_ids[0]
        best_val = logits[best_id]
        for tid in allowed_ids[1:]:
            if logits[tid] > best_val:
                best_val = logits[tid]
                best_id = tid
        return best_id
    arr = np.array(logits, dtype=np.float32)
    return int(allowed_ids[int(np.argmax(arr[allowed_ids]))])


def _pick_from_trie(
    llm: Small_LLM_Model, trie: Trie, input_ids: List[int]
) -> str:
    """Generate tokens constrained by the Trie until a leaf is reached."""
    node = trie.root
    generated: List[int] = []
    while (allowed := trie.allowed_next_ids(node)):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = _mask_logits(logits, allowed)
        node = trie.step(node, next_id)
        generated.append(next_id)
    if node.value is None:
        raise ValueError("Trie generation ended without reaching a leaf")
    return node.value


def _build_trie(
    llm: Small_LLM_Model, prefix: str, entries: Dict[str, str]
) -> Trie:
    """Build a Trie from a {label: value} mapping with a shared prefix."""
    trie = Trie()
    prefix_ids = llm.encode(prefix)[0].tolist()
    for label, value in entries.items():
        full_ids = llm.encode(prefix + label)[0].tolist()
        trie.insert(full_ids[len(prefix_ids):], value)
    return trie


def _unescape(s: str) -> str:
    """Safely unescape a JSON string value."""
    try:
        res = json.loads(f'"{s}"')
        return res if isinstance(res, str) else s
    except json.JSONDecodeError:
        if s.endswith('\\') and not s.endswith('\\\\'):
            s += '\\'
        try:
            res = json.loads(f'"{s}"')
            return res if isinstance(res, str) else s
        except json.JSONDecodeError:
            return s


def _generate_string(
    llm: Small_LLM_Model, vocab: Vocab, input_ids: List[int],
    max_tokens: int = 50,
) -> str:
    """Generate string tokens until an unescaped quote is chosen."""
    generated: List[int] = []
    allowed = vocab.non_quote_ids + vocab.quote_ids
    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = _mask_logits(logits, allowed)
        if next_id in vocab.quote_ids:
            token_str = llm.decode([next_id])
            match = re.search(r'(?<!\\)(?:\\\\)*"', token_str)
            if match:
                prefix = token_str[:match.start()]
                return _unescape(llm.decode(generated) + prefix)
            break
        generated.append(next_id)
    return _unescape(llm.decode(generated))


def _generate_number(
    llm: Small_LLM_Model, vocab: Vocab, input_ids: List[int],
    stop_ids: List[int], max_tokens: int = 15,
) -> str:
    """Generate digit/dot/dash tokens until a stop token."""
    generated: List[int] = []
    allowed = list(vocab.digit_ids) + stop_ids
    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = _mask_logits(logits, allowed)
        if next_id in stop_ids:
            break
        generated.append(next_id)
    return llm.decode(generated).strip()


def _decode_param(
    llm: Small_LLM_Model, vocab: Vocab, schema: Any,
    context: str, input_ids: List[int], bool_trie: Trie,
) -> Any:
    """Decode a single parameter value based on its type."""
    if schema.type == "boolean":
        try:
            return _pick_from_trie(llm, bool_trie, input_ids) == "true"
        except ValueError:
            return False
    if schema.type in ("number", "integer"):
        stop_ids = []
        for ch in (",", "}", "\n", " ", '"'):
            try:
                stop_ids.append(vocab.find_char_token(ch))
            except ValueError:
                pass
        num = _generate_number(llm, vocab, input_ids, stop_ids)
        try:
            return int(float(num)) if schema.type == "integer" else float(num)
        except ValueError:
            return 0 if schema.type == "integer" else 0.0
    if schema.type == "string":
        ids = llm.encode(context + '"')[0].tolist()
        return _generate_string(llm, vocab, ids)
    return None


def _param_context(
    prompt: str, fn_name: str,
    params: Dict[str, Any], param_name: str,
) -> str:
    """Build the JSON context prefix for the next parameter."""
    state = json.dumps({"name": fn_name, "parameters": params})
    truncated = state[:-2]
    if truncated.startswith('{"name":'):
        truncated = '{ "name":' + truncated[8:]
    sep = ", " if params else ""
    return f"User: {prompt}\nJSON: {truncated}{sep}\"{param_name}\": "


def generate_json(
    llm: Small_LLM_Model, vocab: Vocab,
    prompt: str,
    functions: Dict[str, FunctionDefinition],
) -> str:
    """Run fully constrained decoding pipeline for one user query."""
    # Build tries
    fn_trie = _build_trie(
        llm, '\nJSON: { "name": "',
        {name: name for name in functions},
    )
    bool_trie = _build_trie(llm, '": ', {"true": "true", "false": "false"})

    # Select function
    fn_list = [fn.dict() for fn in functions.values()]
    context = (
        f"Available Functions:\n{json.dumps(fn_list, indent=2)}\n\n"
        f"User: {prompt}\n"
        f'JSON: {{ "name": "'
    )
    input_ids = llm.encode(context)[0].tolist()
    try:
        fn_name = _pick_from_trie(llm, fn_trie, input_ids)
    except ValueError:
        fn_name = list(functions.keys())[0]

    # Decode parameters
    params: Dict[str, Any] = {}
    target = functions.get(fn_name)
    if target and target.parameters:
        for pname, schema in target.parameters.items():
            ctx = _param_context(prompt, fn_name, params, pname)
            ids = llm.encode(ctx)[0].tolist()
            params[pname] = _decode_param(
                llm, vocab, schema, ctx, ids, bool_trie
            )

    return json.dumps(
        {"prompt": prompt, "name": fn_name, "parameters": params}, indent=2
    )
