import json
from typing import Any, Dict

from llm_sdk import Small_LLM_Model

from src.constraints import (
    GenerationContext,
    JsonObject,
    build_generation_context,
    pick_from_trie,
    token_ids,
)
from src.parsing import FunctionDefinition
from src.value_decoder import build_param_context, decode_param
from src.vocab import Vocab


def generate_call(
    llm: Small_LLM_Model,
    vocab: Vocab,
    prompt: str,
    context_data: GenerationContext,
) -> JsonObject:
    """Run constrained decoding for one user query."""
    context = (
        f"Available Functions:\n{context_data.functions_block}\n"
        f"User: {prompt}\n"
        f'JSON: {{ "name": "'
    )
    input_ids = token_ids(llm, context)
    fn_name = pick_from_trie(llm, context_data.function_trie, input_ids)
    function = context_data.functions[fn_name]

    params: Dict[str, Any] = {}
    for param_name, schema in function.parameters.items():
        context = build_param_context(prompt, function, params, param_name)
        ids = token_ids(llm, context)
        params[param_name] = decode_param(
            llm, vocab, schema, context, ids, context_data.boolean_trie
        )

    return {"prompt": prompt, "name": fn_name, "parameters": params}


def generate_json(
    llm: Small_LLM_Model,
    vocab: Vocab,
    prompt: str,
    functions: Dict[str, FunctionDefinition],
) -> str:
    """Compatibility wrapper returning one generated call as JSON text."""
    context_data = build_generation_context(llm, functions)
    return json.dumps(
        generate_call(llm, vocab, prompt, context_data),
        indent=2,
    )
