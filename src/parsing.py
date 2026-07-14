"""Load and validate the two input JSON files (prompts and function
definitions) into pydantic models, rejecting empty strings and
duplicate keys along the way."""

import json
from typing import Dict, List, Literal, Tuple

from pydantic import (
    BaseModel,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)


def _no_duplicate_keys(
    pairs: List[Tuple[str, object]],
) -> Dict[str, object]:
    """object_pairs_hook that rejects a JSON object with a repeated
    key (e.g. two parameters with the same name).

    Args:
        pairs: The (key, value) pairs of one JSON object, in the
            order json.loads encountered them.

    Returns:
        The pairs collapsed into a dict, once confirmed key-unique.
    """
    seen: Dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Error: duplicate key: {key!r}")
        seen[key] = value
    return seen


class TypeSchema(BaseModel):
    """The declared type of one parameter or return value.

    Attributes:
        type: One of "number", "integer", "string", "boolean", or
            "null".
        optional: Whether the parameter may be omitted by the
            caller. Defaults to False.
    """

    type: Literal["number", "integer", "string", "boolean", "null"]


class Prompt(BaseModel):
    """A single natural-language prompt from the test file.

    Attributes:
        prompt: The natural-language request text. Must not be
            empty or whitespace-only.
    """

    prompt: str

    @field_validator("prompt")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        """Reject an empty or whitespace-only prompt.

        Args:
            value: The raw "prompt" field value.

        Returns:
            The unchanged value, once confirmed non-empty.
        """
        if not value.strip():
            raise ValueError("Error: prompt must not be empty")
        return value


class FunctionDefinition(BaseModel):
    """One callable function: its name, parameter types, and return
    type.

    Attributes:
        name: The function's unique identifier. Must not be empty.
        description: Human-readable summary of what the function
            does. Must not be empty.
        parameters: Mapping of parameter name to its TypeSchema.
            Parameter names must not be empty.
        returns: The TypeSchema of the function's return value.
    """

    name: str
    description: str
    parameters: Dict[str, TypeSchema]
    returns: TypeSchema

    @field_validator("name", "description")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        """Reject an empty or whitespace-only name/description.

        Args:
            value: The raw "name" or "description" field value.

        Returns:
            The unchanged value, once confirmed non-empty.
        """
        if not value.strip():
            raise ValueError("Error: must not be empty")
        return value

    @model_validator(mode="after")
    def _no_empty_parameter_name(self) -> "FunctionDefinition":
        """Reject a function whose "parameters" object has an empty
        key.

        Args:
            None (validates ``self.parameters``).

        Returns:
            The unchanged instance, once confirmed valid.
        """
        for key in self.parameters:
            if not key.strip():
                raise ValueError(
                    "Error: parameter name must not be empty"
                )
        return self


_PROMPTS = TypeAdapter(List[Prompt])
_FUNCTIONS = TypeAdapter(List[FunctionDefinition])


def _read_text(path: str) -> str:
    """Read a file's contents; I/O failures become ValueError.

    Args:
        path: Path of the file to read.

    Returns:
        The file's full text content.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as error:
        raise ValueError(f"Error: file not found: {path}") from error
    except OSError as error:
        raise ValueError(
            f"Error: cannot read {path}: {error}"
        ) from error


def parse_prompts(path: str) -> List[Prompt]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to a JSON file containing a list of
            ``{"prompt": str}`` objects.

    Returns:
        The parsed list of Prompt objects, in file order.
    """
    raw = _read_text(path)
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
        return _PROMPTS.validate_python(data)
    except (ValidationError, ValueError) as error:
        raise ValueError(
            f"Error: invalid prompts in {path}: {error}"
        ) from error


def parse_functions(path: str) -> Dict[str, FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to a JSON file containing a list of function
            definition objects (name, description, parameters,
            returns).

    Returns:
        A dict keyed by function name for O(1) lookup during
        generation.
    """
    raw = _read_text(path)
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
        definitions = _FUNCTIONS.validate_python(data)
    except (ValidationError, ValueError) as error:
        raise ValueError(
            f"Error: invalid function definitions in {path}: {error}"
        ) from error

    if not definitions:
        raise ValueError(f"Error: no function definitions in {path}")
    functions: Dict[str, FunctionDefinition] = {}
    for function in definitions:
        if function.name in functions:
            raise ValueError(
                f"Error: duplicate function name: {function.name}"
            )
        functions[function.name] = function
    return functions
