"""Load and validate the two input JSON files using pydantic models.

Two files are parsed:
  functions_definition.json — the functions the model can call.
  function_calling_tests.json — the natural-language prompts to process.

Pydantic raises ValidationError for any type or field mismatch, which we
convert to a plain ValueError so the caller gets a clean error message.
"""

import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ValidationError


class TypeSchema(BaseModel):
    """The declared type of one parameter or return value."""

    # Only these four primitive types are supported.
    type: Literal["number", "integer", "string", "boolean"]
    optional: bool = False


class Prompt(BaseModel):
    """A single natural-language prompt from the test file."""

    prompt: str


class FunctionDefinition(BaseModel):
    """One callable function: its name, parameter types, and return type."""

    name: str
    description: str
    parameters: Dict[str, TypeSchema]   # key = parameter name
    returns: TypeSchema


def _load_json_array(path: str) -> List[Any]:
    """Read a JSON file and return its contents, which must be an array."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Error: file not found: {path}") from error
    except OSError as error:
        raise ValueError(f"Error: cannot read {path}: {error}") from error
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
            raise ValueError(f"Error: prompt entry {index} invalid: {error}") from error
    return prompts


def parse_functions(path: str) -> Dict[str, FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Returns a dict keyed by function name for O(1) lookup during generation.
    """
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
            raise ValueError(f"Error: duplicate function name: {function.name}")
        functions[function.name] = function
    if not functions:
        raise ValueError("Error: at least one function definition is required")
    return functions
