*This project has been created as part of the 42 curriculum by tlaghzal.*

## Description

Call Me Maybe is a mandatory 42 project about function calling with a small LLM.
The program receives natural-language prompts and a list of available function
definitions, then writes a JSON array describing which function should be called
and which typed parameters should be passed to it.

Example output object:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The important part is that the model is not asked to freely write JSON. The code
uses constrained decoding so each generated token must keep the output inside
the allowed function names and parameter value types.

## Instructions

### Installation

Install dependencies with:

```bash
make install
```

### Execution

Run the default input files:

```bash
make run
```

Equivalent direct command:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

The output directory is generated at runtime and is ignored by git.

### Debugging

```bash
make debug
```

### Linting

```bash
make lint
```

### Cleaning Up

```bash
make clean
```

## Algorithm Explanation

The decoder follows the subject's mandatory constrained-decoding idea:

1. The program loads and validates both input JSON files with pydantic models.
2. The function definitions are inserted once into a token-id trie. During
   generation, the LLM can only choose token ids that keep at least one valid
   function name path alive.
3. After the function name is selected, parameters are generated one by one using
   the selected function schema.
4. Boolean parameters are constrained to the trie values `true` and `false`.
5. Number and integer parameters are constrained to tokens that keep a valid
   numeric prefix. Stop tokens are only allowed after a complete number.
6. String parameters are generated inside a JSON string context and stop on an
   unescaped quote.
7. The final file is written with `json.dump`, so the produced file is always
   valid JSON and contains only the required keys: `prompt`, `name`, and
   `parameters`.

The function choice comes from the LLM logits under a token constraint. The code
does not choose functions with keyword rules or hardcoded examples.

## Design Decisions

- Pydantic is used for all project classes that hold structured data.
- The code is split by responsibility:
  - `constraints.py` handles token ids, logits masking, tries, and reusable
    generation context.
  - `value_decoder.py` handles schema-specific parameter decoding.
  - `decode.py` keeps only the high-level function-call flow.
- The implementation stays inside the mandatory subject. It does not implement
  bonus tokenizer recoding, model switching, batching, visualization, or nested
  argument support.
- The provided SDK is used through public methods only.
- Errors from missing files, invalid JSON, model initialization, and generation
  are caught and reported clearly.

## File Organization

Detailed file-by-file explanations, including each file's input, output, logic,
and design reason, are in `docs/file_guide.md`.

## Performance Analysis

- JSON validity: the output file is written by Python's JSON module.
- Schema reliability: function names are constrained to declared functions, and
  parameter values are constrained by declared primitive types.
- Accuracy target: the subject asks for 90%+ function and argument accuracy. The
  trie and type constraints improve reliability compared with prompt-only JSON
  generation, while the final quality still depends on the small model logits.
- Speed: prompts are processed sequentially, but shared trie constraints and
  vocabulary token groups are built once and reused for every prompt.

## Challenges Faced & Solutions

- Small models often produce invalid JSON when prompted normally. The solution is
  to never ask the model to freely write the final object.
- Tokenizers can split words and punctuation in surprising ways. The trie is
  built with the SDK's own `encode` method so the allowed paths match the model
  vocabulary.
- Input files may be missing or malformed. Parsing code catches JSON errors and
  validation errors and turns them into readable messages.

## Testing Strategy

Testing should include:

1. `make lint`
2. `make run`
3. Check that `data/output/function_calling_results.json` exists.
4. Validate that the output is a JSON array.
5. Confirm each object contains exactly `prompt`, `name`, and `parameters`.
6. Try malformed JSON and missing file paths to verify clear error messages.

## File Guide

See `docs/file_guide.md` for the complete explanation of every project file and
folder.

## Resources

- Pydantic documentation: https://docs.pydantic.dev/
- Python argparse documentation: https://docs.python.org/3/library/argparse.html
- Python json documentation: https://docs.python.org/3/library/json.html
- Qwen3 model family: https://huggingface.co/Qwen
- AI usage: AI was used to review the subject requirements, simplify the code
  organization, improve wording in this README, and check for lint/type issues.
