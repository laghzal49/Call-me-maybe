"""Call Me Maybe — function-calling engine using constrained decoding.

Module map
----------
- ``parsing``  : load and validate the two input JSON files (pydantic models).
- ``trie``     : Trie data structure for fixed-choice constrained token paths.
- ``decoder``  : Decoder — constrained token-by-token generation (vocab, masking,
                 tries, and the per-prompt run loop all in one place).
- ``output``   : validate generated results against the schema; write JSON.
- ``__main__`` : command-line entry point and orchestration.
"""
