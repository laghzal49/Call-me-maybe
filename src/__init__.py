"""Call Me Maybe — function-calling engine using constrained decoding.

This package turns natural-language prompts into structured function calls
(``{"prompt", "name", "parameters"}``) using a small local LLM whose token
choices are constrained at generation time to guarantee valid, schema-compliant
JSON.

Module map
----------
- ``parsing``       : load + validate the two input JSON files (pydantic models).
- ``trie``          : Trie data structure; encode_ids helper.
- ``vocab``         : map token ids to text; expose per-type allowed token sets.
- ``masking``       : pick_allowed / pick_excluding — logit masking primitives.
- ``context``       : GenerationContext and build_generation_context (built once).
- ``state_machine`` : StateMachine — the token-by-token constrained decoder.
- ``output``        : validate generated results against the schema; write JSON.
- ``__main__``      : command-line entry point and orchestration.
"""
