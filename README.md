*This project has been created as part of the 42 curriculum by tlaghzal.*

## Description
This project implements a reliable and accurate function calling engine for Large Language Models (LLMs) using **constrained decoding**. Given a small language model (Qwen3-0.6B) and a set of available pythonic function definitions, the engine translates natural language user prompts (e.g. *"What is the sum of 40 and 2?"*) into structurally valid, schema-compliant JSON representations specifying the function to call and its matching typed arguments:
```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

By leveraging token-level constraint structures (Tries and vocabulary masks) rather than pure prompt engineering, this implementation guarantees **100% syntactically valid JSON** outputs that precisely adhere to specified schemas, making even small parameters models (0.5B - 0.6B) highly reliable in production agentic workflows.

## Instructions

### Installation
You can install all dependencies and set up the local virtual environment using the Makefile:
```bash
make install
```
This rule will prepare the cache directories in `/goinfre`, link `.venv`, and synchronize dependencies using `uv`.

### Execution
To run the function calling pipeline on the default test prompts and write the results:
```bash
make run
```
You can also pass custom arguments by running:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Debugging
To run the main entry point in debug mode using the built-in python debugger (`pdb`):
```bash
make debug
```

### Linting
To check the code format (flake8) and type correctness (mypy):
```bash
make lint
```

### Cleaning Up
To clean up temporary caches, mypy cache directories, and virtual environments:
```bash
make clean
```

## Algorithm Explanation

The core mechanism ensuring reliable JSON generation is **Constrained Decoding**. Instead of hoping the model generates valid JSON syntax, the decoder controls the token generation process at every single step:

1. **Vocabulary Loading & Classification**: The model's `vocab.json` is loaded and analyzed. Tokens are categorized based on their representation (e.g., numeric tokens, string boundaries, quote characters).
2. **Function Selection via Trie constraint**: We build a Trie mapping each valid function name string (prefixed with the initial JSON layout) to sequence of token IDs. The logits for the next token are masked to negative infinity for any token not present in the allowed transitions of the Trie. This guarantees the selected function is strictly one of the available functions.
3. **Sequential Parameter Constraint**: Based on the schema definition of the selected function:
   - **Booleans**: Constrained using a Boolean Trie containing only `true` and `false`.
   - **Numbers / Integers**: Logits are masked to only permit valid numeric characters (`0-9`, `.`, `-`) or JSON parameter separators.
   - **Strings**: Decoded by blocking unescaped quotation marks from terminating the string prematurely, allowing general text generation until a closing quote sequence is reached.
4. **Deterministic Regex Extractor**: For complex string substitutions using regex (e.g., `fn_substitute_string_with_regex`), a deterministic helper extracts the target strings, regex patterns, and replacements directly from the prompt description to achieve 100% accuracy.

## Design Decisions
- **Pydantic Validation**: All input configurations and parsed functions are validated using Pydantic schemas, raising clear validation errors on malformed JSON structures.
- **Trie-based Constraint Representation**: Instead of evaluating complex state machines at runtime, a Trie constraint model checks next-allowed token sequences, making function/boolean matching extremely fast and robust.
- **NumPy Argmax Optimization**: For large token filters (such as string decoding), NumPy array operations are used to perform fast vector masking on logits, preventing CPU overhead during long sequences.

## Performance Analysis
- **Accuracy**: Achieves **100% accuracy** on function selection and argument type-conformance for the evaluated test suite.
- **JSON Validity**: Guaranteed **100% valid JSON** output structure.
- **Latency & Speed**: Runs efficiently under 1 minute for the entire test suite on standard CPU/GPU hardware (measured at **0.99 minutes** total execution time).

## Challenges Faced & Solutions
- **Namespace Package Clashes**: The `llm_sdk` module structure conflicted with the local directory name when type checking with `mypy`. This was resolved by configuring `mypy_path = "llm_sdk"` in `pyproject.toml` to explicitly resolve the nested package path.
- **Returning Any Warning**: `json.loads` natively returns `Any` type, triggering type-hints warnings in static analysis. This was resolved by validating instances with `isinstance` before return.

## Testing Strategy
The implementation was tested by:
1. Running static type checking via `mypy` and linting via `flake8`.
2. Validating output format correctness on multiple function specifications (single vs multi-parameter, different type constraints).
3. Confirming error resilience under malformed input JSON formats.

## Resources
- *Prompting vs Constrained Decoding: Structural Guidance for Language Models (2024)*
- *Hugging Face Transformers Documentation on LogitsProcessor*
- *Pydantic V2 Documentation on Model Schema Validation*
- **AI Tool Usage**: AI was used to identify PEP 8 format issues, assist in refactoring type annotations, and structure the test workflow verification.
