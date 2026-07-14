"""Load and validate prompts and function definitions from JSON,
rejecting empty strings, duplicate keys, and unknown fields."""

import json
from typing import Dict, List, Literal, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
)

JsonType = Literal["number", "integer", "string", "boolean", "null"]


def _require_non_blank(value: str) -> str:
    """Shared field validator: reject empty/whitespace-only strings.

    Args:
        value: The field value being validated.

    Returns:
        The value unchanged, when it contains visible characters.

    Raises:
        ValueError: If the value is empty or whitespace-only.
    """
    if not value.strip():
        raise ValueError("must not be empty")
    return value


class StrictModel(BaseModel):
    """Base model: unknown keys are a validation error."""

    model_config = ConfigDict(extra="forbid")


class TypeSchema(StrictModel):
    """Declared type of one parameter or return value.

    Attributes:
        type: One of the supported JSON type names.
    """

    type: JsonType


class Prompt(StrictModel):
    """One natural-language prompt.

    Attributes:
        prompt: The request text; must not be blank.
    """

    prompt: str

    _not_blank = field_validator("prompt")(_require_non_blank)


class FunctionDefinition(StrictModel):
    """One callable function: name, parameter types, return type.

    Attributes:
        name: The function name; must not be blank.
        description: Human-readable summary; must not be blank.
        parameters: Parameter type schemas keyed by parameter name.
        returns: The declared return type schema.
    """

    name: str
    description: str
    parameters: Dict[str, TypeSchema]
    returns: TypeSchema

    _not_blank = field_validator("name", "description")(_require_non_blank)

    @field_validator("parameters")
    @classmethod
    def _no_blank_parameter_name(
        cls, value: Dict[str, TypeSchema]
    ) -> Dict[str, TypeSchema]:
        """Reject parameter dicts with empty/whitespace-only names.

        Args:
            value: The parameters mapping being validated.

        Returns:
            The mapping unchanged, when every name is non-blank.

        Raises:
            ValueError: If any parameter name is blank.
        """
        if any(not key.strip() for key in value):
            raise ValueError("parameter name must not be empty")
        return value


_PROMPTS = TypeAdapter(List[Prompt])
_FUNCTIONS = TypeAdapter(List[FunctionDefinition])


def _reject_duplicate_keys(
    pairs: List[Tuple[str, object]],
) -> Dict[str, object]:
    """Build a dict, failing on any repeated key (object_pairs_hook).

    Args:
        pairs: The (key, value) pairs of one JSON object, in order.

    Returns:
        The pairs assembled into a dict.

    Raises:
        ValueError: If the same key appears more than once.
    """
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _load_json(path: str) -> object:
    """Read a file and parse it as JSON.

    Args:
        path: Path of the JSON file to read.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If the file cannot be read, is not valid JSON,
            or contains a duplicate object key.
    """
    try:
        with open(path, encoding="utf-8") as file:
            raw = file.read()
    except OSError as error:
        raise ValueError(f"Error: cannot read {path}: {error}") from error
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as error:
        raise ValueError(f"Error: invalid JSON in {path}: {error}") from error


def parse_prompts(path: str) -> List[Prompt]:
    """Load a JSON list of ``{"prompt": str}`` objects.

    Args:
        path: Path of the prompts JSON file.

    Returns:
        The validated prompts, in file order (possibly empty).

    Raises:
        ValueError: If the file is unreadable, is invalid JSON, or
            does not match the prompt schema.
    """
    try:
        return _PROMPTS.validate_python(_load_json(path))
    except ValidationError as error:
        raise ValueError(
            f"Error: invalid prompts in {path}: {error}"
        ) from error


def parse_functions(path: str) -> Dict[str, FunctionDefinition]:
    """Load a JSON list of function definitions, keyed by name.

    Args:
        path: Path of the function definitions JSON file.

    Returns:
        The validated definitions keyed by function name, in file
        order.

    Raises:
        ValueError: If the file is unreadable, is invalid JSON, does
            not match the schema, is empty, or declares the same
            function name twice.
    """
    try:
        definitions = _FUNCTIONS.validate_python(_load_json(path))
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
                f"Error: duplicate function name: {function.name}"
            )
        functions[function.name] = function
    return functions
