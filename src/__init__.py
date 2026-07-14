"""Call Me Maybe — function-calling engine using constrained decoding.

Module map
----------
- ``parsing``  : load and validate the two input JSON files
  (pydantic models).
- ``vocab``    : Vocab — cached tokenizer encode/decode plus the
  token-id groups used to mask logits.
- ``trie``     : Trie/TrieNode — token-id trie walked by
  Decoder.choose().
- ``decoder``  : Decoder — constrained token-by-token generation.
- ``__main__`` : command-line entry point, orchestration, and JSON
  output.
"""
