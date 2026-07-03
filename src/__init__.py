"""Call Me Maybe — function-calling engine using constrained decoding.

Module map
----------
- ``parsing``  : load and validate the two input JSON files (pydantic models).
- ``decoder``  : Decoder — constrained token-by-token generation.
- ``output``   : validate generated results against the schema; write JSON.
- ``__main__`` : command-line entry point and orchestration.
"""
