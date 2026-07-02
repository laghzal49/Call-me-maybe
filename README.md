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

## How It Works

```mermaid
flowchart LR
    FD[functions_definition.json] --> P[parsing.py]
    TI[function_calling_tests.json] --> P
    P --> D["Decoder setup (once)<br/>vocab token sets · tries ·<br/>functions text block"]
    D --> R["Decoder.run(prompt)<br/>trie-constrained name +<br/>type-constrained values"]
    R --> V["output.py<br/>schema validation"]
    V --> O[function_calling_results.json]
```

At each generation step the model produces logits for every token in the
vocabulary; the decoder sets every invalid token to negative infinity and takes
the argmax of what remains. JSON structure (braces, quotes, keys, commas) is
never generated — it is written directly, and the model is only consulted where
there is a real choice:

```mermaid
flowchart LR
    L["LLM logits"] --> M["mask invalid<br/>tokens to −∞"] --> A[argmax] --> T[token]
    T -->|append & repeat| L
```

- **Function name** — constrained to a trie built from the declared function
  names; the model is only called where names diverge.
- **Boolean** — constrained to a trie of `true` / `false`.
- **Integer / number** — digits, an optional leading minus, at most one dot;
  stopping is itself a constrained choice of an end token.
- **String** — any token without a quote is content; generation stops when a
  quote-bearing token beats the best content token.

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

The decoder follows the subject's mandatory constrained-decoding idea, split
into a one-time **compile phase** and a per-prompt **decode phase**:

1. The program loads and validates both input JSON files with pydantic models.
2. `grammar.py` compiles the schema **once**, before any prompt runs: every
   distinct FSM state (start of a number, after a digit, inside a string, a
   trie branch point) is turned into a boolean mask the width of the
   vocabulary, and every literal, zero-entropy span of the output template
   (`", "parameters": {`, `"key": `, `, `, `}}`, ...) is pre-encoded to token
   ids. Function names are inserted once into a token-id trie; each branching
   node gets its mask cached at compile time too.
3. `decoder.py` only *walks* that compiled grammar. Literal spans are
   appended with no model call. The model is called only where the output
   genuinely branches: which function name, which digits, which string
   characters, `true` vs `false` — and each such call indexes into a mask
   that was already built, instead of rebuilding one.
4. After the function name is selected, parameters are generated one by one using
   the selected function schema.
5. Boolean parameters are constrained to the trie values `true` and `false`.
6. Number and integer parameters are constrained to tokens that keep a valid
   numeric prefix. Stop tokens are only allowed after a complete number.
7. String parameters are generated inside a JSON string context and stop on an
   unescaped quote.
8. The final file is written with `json.dump`, so the produced file is always
   valid JSON and contains only the required keys: `prompt`, `name`, and
   `parameters`.

The function choice comes from the LLM logits under a token constraint. The code
does not choose functions with keyword rules or hardcoded examples.

## Design Decisions

- Pydantic is used for all project classes that hold structured data.
- The code is split by responsibility:
  - `trie.py` — Trie data structure for fixed-choice constrained token paths.
  - `grammar.py` — compile phase: builds every vocabulary mask and every
    pre-encoded literal span once, before any prompt is processed.
  - `decoder.py` — `Decoder` class: probes the model's real vocab width,
    compiles the grammar, and runs the per-prompt constrained generation loop.
  - `parsing.py` / `output.py` — I/O and schema validation.
- The `Decoder` is set up once per run (grammar, tries, functions block) and
  `run(prompt)` is called for each prompt, sharing all precomputed state — no
  mask or literal span is ever rebuilt mid-run.
- The implementation stays inside the mandatory subject. It does not implement
  bonus tokenizer recoding, model switching, batching, visualization, or nested
  argument support.
- The provided SDK is used through public methods only.
- Errors from missing files, invalid JSON, model initialization, and generation
  are caught and reported clearly.

## File Organization

```
src/
├── __main__.py   — CLI entry point and orchestration
├── parsing.py    — pydantic models + JSON input loading
├── trie.py       — token-id trie for constrained name/boolean generation
├── grammar.py    — compile phase: vocab masks + pre-encoded literal spans
├── decoder.py    — Decoder class: compiles the grammar, walks it per prompt
└── output.py     — schema validation + JSON file writing
```

## Performance Analysis

- JSON validity: the output file is written by Python's JSON module.
- Schema reliability: function names are constrained to declared functions, and
  parameter values are constrained by declared primitive types.
- Accuracy target: the subject asks for 90%+ function and argument accuracy. The
  trie and type constraints improve reliability compared with prompt-only JSON
  generation, while the final quality still depends on the small model logits.
- Speed: prompts are processed sequentially, but the grammar (vocab masks,
  trie constraints, literal spans) is compiled exactly once and reused for
  every prompt. Each model call is a full forward pass with no KV cache, so
  the decode loop only ever calls the model where the output genuinely
  branches — never on forced structural tokens.

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

## Resources

- Pydantic documentation: https://docs.pydantic.dev/
- Python argparse documentation: https://docs.python.org/3/library/argparse.html
- Python json documentation: https://docs.python.org/3/library/json.html
- Qwen3 model family: https://huggingface.co/Qwen
- AI usage: AI was used to review the subject requirements, simplify the code
  organization, improve wording in this README, and check for lint/type issues.
