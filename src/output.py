"""Validate generated calls against the schema and write the JSON output file."""

import json
import os
from typing import Any, Dict, List

from src.parsing import FunctionDefinition

JsonObject = Dict[str, Any]


def validate_result(
    result: JsonObject,
    functions: Dict[str, FunctionDefinition],
) -> None:
    """Raise ValueError if a result breaks the required schema.

    Checks (in order):
      1. Exactly the three required keys: prompt, name, parameters.
      2. The function name exists in the definitions.
      3. The parameter keys match the function's declared parameters exactly.
      4. Each value's Python type matches its declared JSON type.
    """
    # Exactly three keys — no extras, no missing.
    if set(result) != {"prompt", "name", "parameters"}:
        raise ValueError(f"unexpected keys: {sorted(result)}")

    name = result["name"]
    if name not in functions:
        raise ValueError(f"unknown function: {name}")

    function = functions[name]
    params = result["parameters"]

    # Parameter set must match exactly — no extras, no missing.
    if set(params) != set(function.parameters):
        raise ValueError(f"{name}: parameter mismatch: {sorted(params)}")

    # Check each value's Python type against its declared schema type.
    for key, schema in function.parameters.items():
        value = params[key]
        if schema.type == "string" and not isinstance(value, str):
            raise ValueError(f"{name}.{key}: expected string")
        if schema.type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{name}.{key}: expected boolean")
        # bool is a subclass of int in Python, so we must reject it explicitly
        # for numeric types (True / False must not pass as 1 / 0).
        if schema.type in {"number", "integer"} and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{name}.{key}: expected {schema.type}")


def write_results(path: str, results: List[JsonObject]) -> None:
    """Write the results list as pretty-printed JSON.

    Creates the output directory if it does not exist.
    Using json.dump (not manual string building) guarantees the file is
    always valid JSON regardless of what values the model produced.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")   # trailing newline for POSIX compliance
