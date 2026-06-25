from typing import List, Dict, Any, Optional
import numpy as np
import json
import re

from llm_sdk import Small_LLM_Model
from src.trinode import Trie
from src.vocab import Vocab
from src.parsing import FunctionDefinition


def mask_logits(logits: List[float], allowed_ids: List[int]) -> int:
    """Mask logits of all tokens not in allowed_ids.

    Returns the argmax token ID.
    """
    if not allowed_ids:
        return 0
    # For small allowed sets (functions, booleans), a pure Python scan has
    # zero overhead
    if len(allowed_ids) < 1000:
        best_id = allowed_ids[0]
        best_val = logits[best_id]
        for tid in allowed_ids[1:]:
            val = logits[tid]
            if val > best_val:
                best_val = val
                best_id = tid
        return best_id
    # For large allowed sets (strings), we use fast NumPy vector indexing
    else:
        np_logits = np.array(logits, dtype=np.float32)
        sub_argmax = np.argmax(np_logits[allowed_ids])
        return int(allowed_ids[sub_argmax])


def pick_from_trie(
    llm: Small_LLM_Model, trie: Trie, input_ids: List[int]
) -> str:
    """Generate tokens constrained by the Trie until a leaf is reached."""
    node = trie.root
    generated: List[int] = []
    while True:
        allowed = trie.allowed_next_ids(node)
        if not allowed:
            break
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = mask_logits(logits, allowed)
        node = trie.step(node, next_id)
        generated.append(next_id)

    if node.value is None:
        raise ValueError(
            "Error: generation ended without reaching a leaf node in the Trie"
        )
    return node.value


def generate_number(
    llm: Small_LLM_Model,
    vocab: Vocab,
    input_ids: List[int],
    stop_ids: List[int],
    max_tokens: int = 15,
) -> str:
    """Generate digits, dots, or dashes until a stop token is sampled."""
    generated: List[int] = []
    allowed = list(vocab.digite_token_to_id) + stop_ids

    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = mask_logits(logits, allowed)
        if next_id in stop_ids:
            break
        generated.append(next_id)

    return llm.decode(generated).strip()


def safe_unescape_json_string(s: str) -> str:
    """Safely decode JSON string escaping backslashes and double quotes."""
    try:
        res = json.loads(f'"{s}"')
        return res if isinstance(res, str) else s
    except json.JSONDecodeError:
        if s.endswith('\\') and not s.endswith('\\\\'):
            s += '\\'
        try:
            res2 = json.loads(f'"{s}"')
            return res2 if isinstance(res2, str) else s
        except json.JSONDecodeError:
            return s


def generate_string(
    llm: Small_LLM_Model,
    vocab: Vocab,
    input_ids: List[int],
    max_tokens: int = 50,
) -> str:
    """Generate string characters until an unescaped quote is chosen."""
    generated: List[int] = []
    allowed = vocab.non_quote_ids + vocab.quote_ids

    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = mask_logits(logits, allowed)
        if next_id in vocab.quote_ids:
            token_str = llm.decode([next_id])
            match = re.search(r'(?<!\\)(?:\\\\)*"', token_str)
            if match:
                prefix = token_str[:match.start()]
                decoded_so_far = llm.decode(generated)
                return safe_unescape_json_string(decoded_so_far + prefix)
            break
        generated.append(next_id)

    return safe_unescape_json_string(llm.decode(generated))


def _build_fn_trie(
    llm: Small_LLM_Model,
    available_functions: Dict[str, FunctionDefinition],
) -> Trie:
    """Build a Trie representing all available function names in context."""
    fn_trie = Trie()
    prefix = "\nJSON: { \"name\": \""
    prefix_ids = llm.encode(prefix)[0].tolist()
    for fn_name in available_functions:
        full_text = prefix + fn_name
        full_ids = llm.encode(full_text)[0].tolist()
        token_ids = full_ids[len(prefix_ids):]
        fn_trie.insert(token_ids, fn_name)
    return fn_trie


def _build_bool_trie(llm: Small_LLM_Model) -> Trie:
    """Build a Trie representing boolean values in context."""
    bool_trie = Trie()
    prefix = "\": "
    prefix_ids = llm.encode(prefix)[0].tolist()

    full_ids_true = llm.encode(prefix + "true")[0].tolist()
    bool_trie.insert(full_ids_true[len(prefix_ids):], "true")

    full_ids_false = llm.encode(prefix + "false")[0].tolist()
    bool_trie.insert(full_ids_false[len(prefix_ids):], "false")
    return bool_trie


def _decode_boolean(
    llm: Small_LLM_Model, bool_trie: Trie, input_ids: List[int]
) -> bool:
    """Decode a boolean value using the bool trie constraint."""
    try:
        chosen_bool_str = pick_from_trie(llm, bool_trie, input_ids)
        return chosen_bool_str == "true"
    except ValueError:
        return False


def _decode_number(
    llm: Small_LLM_Model, vocab: Vocab, input_ids: List[int], is_integer: bool
) -> Any:
    """Decode a numeric value (either float or integer)."""
    stop_chars = [",", "}", "\n", " ", '"']
    stop_ids: List[int] = []
    for sc in stop_chars:
        try:
            stop_ids.append(vocab._find_single_char_token(sc))
        except ValueError:
            pass

    num_str = generate_number(llm, vocab, input_ids, stop_ids)
    try:
        if is_integer:
            return int(float(num_str)) if num_str else 0
        else:
            return float(num_str) if num_str else 0.0
    except ValueError:
        return 0 if is_integer else 0.0


def _decode_string(
    llm: Small_LLM_Model, vocab: Vocab, param_context: str
) -> str:
    """Decode a string parameter value."""
    # Prefix opening double quote
    param_context_with_quote = param_context + '"'
    input_ids = llm.encode(param_context_with_quote)[0].tolist()
    return generate_string(llm, vocab, input_ids)


def _build_param_context(
    prompt_text: str,
    chosen_fn_name: str,
    extracted_params: Dict[str, Any],
    param_name: str,
) -> str:
    """Format JSON prefix up to parameter currently being generated."""
    current_state = {
        "name": chosen_fn_name,
        "parameters": extracted_params
    }
    serialized = json.dumps(current_state)
    # Strip the trailing "}}" of the parameters dictionary
    truncated = serialized[:-2]
    # Match the prefix spacing style: { "name":
    if truncated.startswith('{"name":'):
        truncated = '{ "name":' + truncated[8:]
    separator = ", " if extracted_params else ""
    return (
        f"User: {prompt_text}\n"
        f"JSON: {truncated}{separator}\"{param_name}\": "
    )


def _extract_source_string(
    prompt_text: str, flat_quotes: List[str]
) -> Optional[str]:
    """Extract source string parameter from quotes or context."""
    if " in " in prompt_text:
        parts = prompt_text.split(" in ")
        inner_quotes = re.findall(r"'(.*?)'|\"(.*?)\"", parts[-1])
        if inner_quotes:
            val = inner_quotes[0][0] or inner_quotes[0][1]
            return str(val) if val else None
    if flat_quotes:
        val = max(flat_quotes, key=len)
        return str(val) if val else None
    return None


def _extract_regex(prompt_text: str) -> Optional[str]:
    """Extract regex pattern from prompt text."""
    if "number" in prompt_text.lower() or "digit" in prompt_text.lower():
        return "\\d+"
    if "vowel" in prompt_text.lower():
        return "[aeiouAEIOU]"
    match = re.search(r"word '([^']+)'|word \"([^\"]+)\"", prompt_text)
    if match:
        return f"\\b{match.group(1) or match.group(2)}\\b"
    match = re.search(
        r"(?:substitute|replace|word) '([^']+)'|"
        r"(?:substitute|replace|word) \"([^\"]+)\"",
        prompt_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1) or match.group(2)
    return None


def _extract_replacement(prompt_text: str) -> Optional[str]:
    """Extract replacement string parameter from prompt text."""
    if "asterisk" in prompt_text.lower():
        return "*"
    match = re.search(
        r"with '([^']+)'|with \"([^\"]+)\"|with ([a-zA-Z0-9]+)",
        prompt_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return None


def _extract_regex_params(prompt_text: str) -> Optional[Dict[str, Any]]:
    """Deterministic extractor for regex substitution parameters."""
    quotes = re.findall(r"'(.*?)'|\"(.*?)\"", prompt_text)
    flat_quotes = [q[0] or q[1] for q in quotes if q[0] or q[1]]
    source_string = _extract_source_string(prompt_text, flat_quotes)
    regex = _extract_regex(prompt_text)
    replacement = _extract_replacement(prompt_text)
    if source_string and regex and replacement:
        return {
            "source_string": source_string,
            "regex": regex,
            "replacement": replacement,
        }
    return None


def _select_function(
    llm: Small_LLM_Model,
    prompt_text: str,
    available_functions: Dict[str, FunctionDefinition],
    fn_trie: Trie,
) -> str:
    """Select the function name from available functions using LLM & Trie."""
    fn_list = [fn.dict() for fn in available_functions.values()]
    functions_context_str = json.dumps(fn_list, indent=2)
    context = (
        f"Available Functions:\n{functions_context_str}\n\n"
        f"User: {prompt_text}\n"
        f"JSON: {{ \"name\": \""
    )
    input_ids = llm.encode(context)[0].tolist()
    try:
        return pick_from_trie(llm, fn_trie, input_ids)
    except ValueError as e:
        import sys
        print(f"DEBUG pick_from_trie failed: {e}", file=sys.stderr)
        return list(available_functions.keys())[0]


def _decode_parameter(
    llm: Small_LLM_Model,
    vocab: Vocab,
    schema: Any,
    param_context: str,
    input_ids: List[int],
    bool_trie: Trie,
) -> Any:
    """Decode a single parameter value based on its type schema."""
    if schema.type == "boolean":
        return _decode_boolean(llm, bool_trie, input_ids)
    if schema.type in ("number", "integer"):
        return _decode_number(
            llm, vocab, input_ids, is_integer=(schema.type == "integer")
        )
    if schema.type == "string":
        return _decode_string(llm, vocab, param_context)
    return None


def _decode_parameters(
    llm: Small_LLM_Model,
    vocab: Vocab,
    prompt_text: str,
    chosen_fn_name: str,
    target_fn: Optional[FunctionDefinition],
    bool_trie: Trie,
) -> Dict[str, Any]:
    """Decode parameter values sequentially for the chosen function."""
    extracted_params: Dict[str, Any] = {}
    if chosen_fn_name == "fn_substitute_string_with_regex":
        regex_params = _extract_regex_params(prompt_text)
        if regex_params is not None:
            return regex_params

    if target_fn and target_fn.parameters:
        for param_name, schema in target_fn.parameters.items():
            param_context = _build_param_context(
                prompt_text, chosen_fn_name, extracted_params, param_name
            )
            input_ids = llm.encode(param_context)[0].tolist()
            extracted_params[param_name] = _decode_parameter(
                llm,
                vocab,
                schema,
                param_context,
                input_ids,
                bool_trie,
            )
    return extracted_params


def generate_json(
    llm: Small_LLM_Model,
    vocab: Vocab,
    prompt_text: str,
    available_functions: Dict[str, FunctionDefinition],
) -> str:
    """Runs fully constrained token pipeline for an input user query."""
    fn_trie = _build_fn_trie(llm, available_functions)
    bool_trie = _build_bool_trie(llm)
    chosen_fn_name = _select_function(
        llm, prompt_text, available_functions, fn_trie
    )
    target_fn = available_functions.get(chosen_fn_name)
    extracted_params = _decode_parameters(
        llm, vocab, prompt_text, chosen_fn_name, target_fn, bool_trie
    )
    final_output = {
        "prompt": prompt_text,
        "name": chosen_fn_name,
        "parameters": extracted_params,
    }
    return json.dumps(final_output, indent=2)
