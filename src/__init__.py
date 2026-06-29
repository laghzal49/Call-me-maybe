"""Call Me Maybe — function-calling engine using constrained decoding.

This package turns natural-language prompts into structured function calls
(``{"prompt", "name", "parameters"}``) using a small local LLM whose token
choices are constrained at generation time to guarantee valid, schema-compliant
JSON.

Module map
----------
- ``parsing``     : load + validate the two input JSON files (pydantic models).
- ``vocab``       : map token ids to their decoded text and expose, for each
                    value type, the set of token ids that are legal to emit.
- ``constraints`` : the ``Trie`` data structure and the ``GenerationContext``
                    that bundles everything precomputed once per run.
- ``decoder``     : the constrained greedy decoding loop (pick function name,
                    then decode each parameter according to its type).
- ``output``      : validate generated results against the schema and write the
                    final JSON output file.
- ``__main__``    : command-line entry point and orchestration.
"""
