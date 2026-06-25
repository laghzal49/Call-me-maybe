import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ValidationError


class TypeSchema(BaseModel):
    """Schema describing the expected type of a parameter or return value."""

    type: Literal["number", "integer", "string", "boolean"]


class Promt(BaseModel):
    """A single natural-language prompt."""

    prompt: str


class FunctionDefinition(BaseModel):
    """Definition of a callable function: name, args, return type."""

    name: str
    description: str
    parameters: Dict[str, TypeSchema]
    returns: TypeSchema


def _load_json_array(path: str) -> List[Any]:
    """Read a JSON file and ensure it's an array."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Error: {path} must contain a JSON array")
    return data


def parse_prompts(path: str) -> List[Promt]:
    """Load and validate prompts from a JSON file."""
    entries = _load_json_array(path)
    prompts: List[Promt] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Error: entry {i} is not a JSON object")
        try:
            prompts.append(Promt(**entry))
        except ValidationError as e:
            raise ValueError(f"Error: entry {i} invalid: {e}")
    return prompts


def parse_functions(path: str) -> Dict[str, FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    entries = _load_json_array(path)
    functions: Dict[str, FunctionDefinition] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Error: entry {i} is not a JSON object")
        try:
            fn = FunctionDefinition(**entry)
        except ValidationError as e:
            raise ValueError(f"Error: entry {i} invalid: {e}")
        if fn.name in functions:
            raise ValueError(f"Error: Duplicate Function Name: {fn.name}")
        functions[fn.name] = fn
    return functions
