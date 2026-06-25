import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ValidationError


class TypeSchema(BaseModel):
    """Schema describing the expected type of a parameter or return value."""

    type: Literal["number", "integer", "string", "boolean"]


class Promt(BaseModel):
    """A single natural-language prompt to be parsed by the LLM."""

    prompt: str


class FunctionDefinition(BaseModel):
    """Definition of a callable function: its name, args, return type."""

    name: str
    description: str
    parameters: Dict[str, TypeSchema]
    returns: TypeSchema


class Promt_parse:
    """Loads and validates a list of prompts from a JSON file."""

    def __init__(self, file_name: str) -> None:
        self.file_name: str = file_name
        self.promte: List[Promt] = []

    def open_file(self) -> List[Any]:
        """Read and JSON-decode the prompts file."""
        try:
            with open(self.file_name) as file:
                all_promte = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: File Not Found: {self.file_name}"
            )
        except PermissionError:
            raise PermissionError(
                f"Error: Permission Denied: {self.file_name}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error: Invalid JSON in {self.file_name}: {e}"
            )
        except OSError as e:
            raise OSError(f"Error: OSError on {self.file_name}: {e}")
        if not isinstance(all_promte, list):
            raise ValueError(
                f"Error: {self.file_name} must contain a JSON array"
            )
        return all_promte

    def start_parse(self) -> None:
        """Parse and validate every prompt entry."""
        promotes = self.open_file()
        for i, entry in enumerate(promotes):
            self._validate_and_append_prompt(entry, i)

    def _validate_and_append_prompt(self, entry: Any, index: int) -> None:
        """Validate a single prompt entry and append it to self.promte."""
        try:
            if not isinstance(entry, dict):
                raise ValueError(f"entry {index} is not a JSON object")
            pr = Promt(prompt=entry["prompt"])
        except KeyError:
            raise ValueError(
                f"Error: entry {index} is missing the 'prompt' key"
            )
        except ValidationError as e:
            raise ValueError(
                f"Error: entry {index} has an invalid 'prompt': {e}"
            )
        self.promte.append(pr)


class Function_parse:
    """Loads and validates available function definitions from a JSON file."""

    def __init__(self, file_name: str) -> None:
        self.file_name: str = file_name
        self.functions_dict: Dict[str, FunctionDefinition] = {}

    def open_file(self) -> List[Any]:
        """Read and JSON-decode the function definitions file."""
        try:
            with open(self.file_name) as file:
                all_promte = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: File Not Found: {self.file_name}"
            )
        except PermissionError:
            raise PermissionError(
                f"Error: Permission Denied: {self.file_name}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error: Invalid JSON in {self.file_name}: {e}"
            )
        except OSError as e:
            raise OSError(f"Error: OSError on {self.file_name}: {e}")
        if not isinstance(all_promte, list):
            raise ValueError(
                f"Error: {self.file_name} must contain a JSON array"
            )
        return all_promte

    def start_parse(self) -> None:
        """
        Parse, validate, and index every function definition.
        """
        promte = self.open_file()
        for i, entry in enumerate(promte):
            self._validate_and_index_function(entry, i)

    def _validate_and_index_function(self, entry: Any, index: int) -> None:
        """Validate a single function entry and index it in dict."""
        try:
            if not isinstance(entry, dict):
                raise ValueError(f"entry {index} is not a JSON object")
            fn_object = FunctionDefinition(**entry)
        except TypeError as e:
            raise ValueError(
                f"Error: entry {index} has invalid fields: {e}"
            )
        except ValidationError as e:
            raise ValueError(
                f"Error: entry {index} failed validation: {e}"
            )
        if fn_object.name in self.functions_dict:
            raise ValueError(
                f"Error: Duplicate Function Name: {fn_object.name}"
            )
        self.functions_dict[fn_object.name] = fn_object
