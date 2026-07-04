from typing import Dict, List, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError


class TypeSchema(BaseModel):
    """The declared type of one parameter or return value."""

    type: Literal["number", "integer", "string", "boolean"]
    optional: bool = False


class Prompt(BaseModel):
    """A single natural-language prompt from the test file."""

    prompt: str


class FunctionDefinition(BaseModel):
    """One callable function: its name, parameter types, and return type."""

    name: str
    description: str
    parameters: Dict[str, TypeSchema]
    returns: TypeSchema


_PROMPTS = TypeAdapter(List[Prompt])
_FUNCTIONS = TypeAdapter(List[FunctionDefinition])


def _read_text(path: str) -> str:
    """Read a file's contents; I/O failures become ValueError."""
    try:
        with open(path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as error:
        raise ValueError(f"Error: file not found: {path}") from error
    except OSError as error:
        raise ValueError(f"Error: cannot read {path}: {error}") from error


def parse_prompts(path: str) -> List[Prompt]:
    """Load and validate prompts from a JSON file."""
    raw = _read_text(path)
    try:
        return _PROMPTS.validate_json(raw)
    except ValidationError as error:
        raise ValueError(
            f"Error: invalid prompts in {path}: {error}") from error


def parse_functions(path: str) -> Dict[str, FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Returns a dict keyed by function name for O(1) lookup during generation.
    """
    raw = _read_text(path)
    try:
        definitions = _FUNCTIONS.validate_json(raw)
    except ValidationError as error:
        raise ValueError(
            f"Error: invalid function definitions in {path}: {error}"
        ) from error

    if not definitions:
        raise ValueError(f"Error: no function definitions in {path}")
    functions: Dict[str, FunctionDefinition] = {}
    for function in definitions:
        if function.name in functions:
            raise ValueError(
                f"Error: duplicate function name: {function.name}")
        functions[function.name] = function
    return functions
