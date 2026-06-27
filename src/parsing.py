import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ValidationError


class TypeSchema(BaseModel):
    """Schema describing the expected type of a parameter or return value."""

    type: Literal["number", "integer", "string", "boolean"]


class Prompt(BaseModel):
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
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Error: invalid JSON in {path}: {error}") from error
    if not isinstance(data, list):
        raise ValueError(f"Error: {path} must contain a JSON array")
    return data


def parse_prompts(path: str) -> List[Prompt]:
    """Load and validate prompts from a JSON file."""
    entries = _load_json_array(path)
    prompts: List[Prompt] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Error: prompt entry {index} is not an object")
        try:
            prompts.append(Prompt(**entry))
        except ValidationError as error:
            raise ValueError(
                f"Error: prompt entry {index} invalid: {error}"
            ) from error
    return prompts


def parse_functions(path: str) -> Dict[str, FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    entries = _load_json_array(path)
    functions: Dict[str, FunctionDefinition] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Error: function entry {index} is not an object")
        try:
            function = FunctionDefinition(**entry)
        except ValidationError as error:
            raise ValueError(
                f"Error: function entry {index} invalid: {error}"
            ) from error
        if function.name in functions:
            raise ValueError(
                f"Error: duplicate function name: {function.name}"
            )
        functions[function.name] = function
    if not functions:
        raise ValueError("Error: at least one function definition is required")
    return functions
