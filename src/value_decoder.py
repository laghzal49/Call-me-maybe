import json
import re
from typing import Any, Dict, List, Literal

from llm_sdk import Small_LLM_Model

from src.constraints import mask_logits, pick_from_trie, token_ids
from src.parsing import FunctionDefinition, TypeSchema
from src.trinode import Trie
from src.vocab import Vocab


def _unescape(value: str) -> str:
    """Safely unescape a JSON string value."""
    try:
        result = json.loads(f'"{value}"')
        return result if isinstance(result, str) else value
    except json.JSONDecodeError:
        if value.endswith("\\") and not value.endswith("\\\\"):
            value += "\\"
        try:
            result = json.loads(f'"{value}"')
            return result if isinstance(result, str) else value
        except json.JSONDecodeError:
            return value


def _generate_string(
    llm: Small_LLM_Model,
    vocab: Vocab,
    input_ids: List[int],
    max_tokens: int = 50,
) -> str:
    """Generate string tokens until an unescaped quote is selected."""
    generated: List[int] = []
    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        next_id = mask_logits(logits, vocab.string_ids)
        if next_id in vocab.quote_id_set:
            token_str = llm.decode([next_id])
            match = re.search(r'(?<!\\)(?:\\\\)*"', token_str)
            if match:
                prefix = token_str[:match.start()]
                return _unescape(llm.decode(generated) + prefix)
            break
        generated.append(next_id)
    return _unescape(llm.decode(generated))


def _is_number_prefix(
    value: str,
    schema_type: Literal["number", "integer"],
) -> bool:
    """Return True when value can still become a valid JSON number."""
    if schema_type == "integer":
        if value in ("", "-"):
            return True
        raw = value[1:] if value.startswith("-") else value
        return raw.isdigit()
    if value in ("", "-", ".", "-."):
        return True
    if value.count(".") > 1:
        return False
    raw = value[1:] if value.startswith("-") else value
    return raw.replace(".", "", 1).isdigit()


def _is_number(value: str, schema_type: Literal["number", "integer"]) -> bool:
    """Return True when value is a complete valid JSON number."""
    if schema_type == "integer":
        return _is_number_prefix(value, schema_type) and value not in ("", "-")
    if value in ("", "-", ".", "-."):
        return False
    return _is_number_prefix(value, schema_type)


def _allowed_number_ids(
    vocab: Vocab,
    current: str,
    schema_type: Literal["number", "integer"],
) -> List[int]:
    """Return token ids that keep a valid numeric prefix."""
    allowed: List[int] = []
    for token_id, token_text in vocab.numeric_tokens:
        candidate = current + token_text
        if _is_number_prefix(candidate, schema_type):
            allowed.append(token_id)
    if _is_number(current, schema_type):
        allowed.extend(vocab.number_stop_ids)
    return allowed


def _generate_number(
    llm: Small_LLM_Model,
    vocab: Vocab,
    input_ids: List[int],
    stop_ids: List[int],
    schema_type: Literal["number", "integer"],
    max_tokens: int = 15,
) -> str:
    """Generate a schema-valid number prefix until a stop token."""
    generated: List[int] = []
    current = ""
    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids + generated)
        allowed = _allowed_number_ids(vocab, current, schema_type)
        next_id = mask_logits(logits, allowed)
        if next_id in stop_ids:
            break
        generated.append(next_id)
        current += vocab.id_to_token[next_id].lstrip("Ġ ")
    return current


def build_param_context(
    prompt: str,
    function: FunctionDefinition,
    params: Dict[str, Any],
    param_name: str,
) -> str:
    """Build the model context for extracting one parameter."""
    schema = function.model_dump()
    previous = json.dumps(params)
    return (
        "Extract one argument for the selected function.\n"
        f"Function definition:\n{json.dumps(schema, separators=(',', ':'))}\n"
        f"User request: {prompt}\n"
        f"Already extracted parameters: {previous}\n"
        f'Selected function: {function.name}\n'
        f'JSON parameter "{param_name}": '
    )


def decode_param(
    llm: Small_LLM_Model,
    vocab: Vocab,
    schema: TypeSchema,
    context: str,
    input_ids: List[int],
    bool_trie: Trie,
) -> Any:
    """Decode one parameter value according to its declared schema type."""
    if schema.type == "boolean":
        return pick_from_trie(llm, bool_trie, input_ids) == "true"
    if schema.type in ("number", "integer"):
        num = _generate_number(
            llm,
            vocab,
            input_ids,
            vocab.number_stop_ids,
            schema.type,
        )
        try:
            return int(float(num)) if schema.type == "integer" else float(num)
        except ValueError:
            return 0 if schema.type == "integer" else 0.0
    if schema.type == "string":
        ids = token_ids(llm, context + '"')
        return _generate_string(llm, vocab, ids)
    return None
