"""Everything precomputed once per run and shared by every prompt."""

from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model

from src.parsing import FunctionDefinition
from src.trie import Trie

# header the model sees before the JSON it must complete
INSTRUCTION = (
    "You convert the request into a function call.\n"
    "Available functions:\n{block}\n"
    "Request: {prompt}\n"
)


def build_functions_block(functions: Dict[str, FunctionDefinition]) -> str:
    """Render the functions as short one-line signatures for the prompt."""
    lines: List[str] = []
    for fn in functions.values():
        # e.g. "- fn_add_numbers(a: number, b: number): Add two numbers..."
        params = ", ".join(f"{n}: {s.type}" for n, s in fn.parameters.items())
        lines.append(f"- {fn.name}({params}): {fn.description}")
    return "\n".join(lines)


class GenerationContext(BaseModel):
    """Precomputed inputs reused for every prompt (built once)."""

    # Trie objects are not pydantic types, so allow arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: Dict[str, FunctionDefinition]
    functions_block: str    # text injected into every prompt
    function_trie: Trie     # constrains the function-name choice
    boolean_trie: Trie      # constrains booleans to true / false


def build_generation_context(
    llm: Small_LLM_Model,
    functions: Dict[str, FunctionDefinition],
) -> GenerationContext:
    """Precompute the functions block and the name / boolean tries."""
    return GenerationContext(
        functions=functions,
        functions_block=build_functions_block(functions),
        function_trie=Trie.from_strings(llm, list(functions)),
        boolean_trie=Trie.from_strings(llm, ["true", "false"]),
    )
