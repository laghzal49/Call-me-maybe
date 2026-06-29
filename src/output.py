"""Validate generated calls against the schema and write the JSON file."""

import json
import os
from typing import Any, Dict, List

from src.parsing import FunctionDefinition

JsonObject = Dict[str, Any]


def validate_result(
    result: JsonObject,
    functions: Dict[str, FunctionDefinition],
) -> None:
    """Raise ValueError if a result breaks the schema."""
    # exactly the three required keys, nothing more
    if set(result) != {"prompt", "name", "parameters"}:
        raise ValueError(f"unexpected keys: {sorted(result)}")
    name = result["name"]
    # the chosen function must exist
    if name not in functions:
        raise ValueError(f"unknown function: {name}")
    function = functions[name]
    params = result["parameters"]
    # the params must match the function's declared params exactly
    if set(params) != set(function.parameters):
        raise ValueError(f"{name}: parameter mismatch: {sorted(params)}")
    # each value's python type must match its declared type
    for key, schema in function.parameters.items():
        value = params[key]
        if schema.type == "string" and not isinstance(value, str):
            raise ValueError(f"{name}.{key}: expected string")
        if schema.type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{name}.{key}: expected boolean")
        # bool is a subclass of int, so reject it for numeric types
        if schema.type in {"number", "integer"} and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{name}.{key}: expected {schema.type}")


def write_results(path: str, results: List[JsonObject]) -> None:
    """Write the results array as pretty JSON, creating the folder if needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    # context manager closes the file even on error
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
