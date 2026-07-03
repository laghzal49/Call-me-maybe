"""Validate each generated call against its function schema, then write JSON.

Constrained decoding should already produce valid results; this is the final
safety net required by the subject (V.4.2): exact keys, a known function name,
exactly the declared parameters, and matching value types.
"""

import json
import os
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError

from src.parsing import FunctionDefinition

JsonObject = Dict[str, Any]

_PYTHON_TYPES: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "integer": (int, float),
    "number": (int, float),
}


class Result(BaseModel):
    """The required output shape: exactly prompt, name, and parameters."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, Any]


def validate_result(
    result: JsonObject,
    functions: Dict[str, FunctionDefinition],
) -> None:
    """Raise ValueError if a result breaks the required schema."""
    try:
        checked = Result.model_validate(result)
    except ValidationError as error:
        raise ValueError(str(error)) from error

    function = functions.get(checked.name)
    if function is None:
        raise ValueError(f"unknown function name: {checked.name}")

    if set(checked.parameters) != set(function.parameters):
        raise ValueError(
            f"{checked.name}: expected parameters "
            f"{sorted(function.parameters)}, got {sorted(checked.parameters)}"
        )

    for key, schema in function.parameters.items():
        value = checked.parameters[key]
        if isinstance(value, bool) and schema.type != "boolean":
            raise ValueError(f"{checked.name}.{key}: expected {schema.type}")
        if not isinstance(value, _PYTHON_TYPES[schema.type]):
            raise ValueError(f"{checked.name}.{key}: expected {schema.type}")


def write_results(path: str, results: List[JsonObject]) -> None:
    """Write the results list as pretty-printed JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
