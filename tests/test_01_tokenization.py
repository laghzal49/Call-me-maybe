"""
CONCEPT 1 — TOKENIZATION
=========================
A tokenizer splits text into subword units called TOKENS and maps each
to a unique integer called a TOKEN ID that the model understands.

    "hello world"  →  ["hello", " world"]  →  [9906, 1917]

Why subwords and not characters or words?
  - Characters: too many steps per word, context window fills up fast.
  - Words: unknown words can't be handled ("ChatGPT" wasn't in old dicts).
  - Subwords (BPE): balance — common words stay whole, rare words split.

Why does this matter for constrained decoding?
  A function name like "fn_add_numbers" may split into SEVERAL tokens:
      ["fn", "_add", "_numbers"]
  The model picks them ONE AT A TIME.  We must constrain each token
  individually — that is why choose() filters token paths (see test_09).

Run: uv run python tests/test_01_tokenization.py
     (requires the model to be downloaded)
"""

import sys

sys.path.insert(0, ".")

from llm_sdk import Small_LLM_Model  # noqa: E402

llm = Small_LLM_Model()

# ── 1. encode: text → token ids ──────────────────────────────────────────────
# encode() returns a 2-D tensor of shape [1, n_tokens].
# .tolist()[0] flattens it to a plain Python list of ints.

text = "fn_add_numbers"
ids = llm.encode(text).tolist()[0]
print(f"text   : {text!r}")
print(f"ids    : {ids}")
# Example output: [14176, 55688, 34062]  (actual ids depend on the tokenizer)

# ── 2. decode: token ids → text ──────────────────────────────────────────────
# decode() reconstructs the original text from a list of token ids.

back = llm.decode(ids)
print(f"decoded: {back!r}")
assert back == text, "encode → decode must be a perfect round-trip"
print("OK: round-trip is exact")

# ── 3. one token at a time ───────────────────────────────────────────────────
# The model sees the SEQUENCE so far and predicts ONE next token.
# We feed it a growing list of ids — that is self.ids in the Decoder.

print()
for i, tok_id in enumerate(ids):
    piece = llm.decode([tok_id])
    print(f"  token {i}: id={tok_id:6d}  text={piece!r}")

# ── 4. why multi-token words need path filtering ─────────────────────────────────────
# If "fn_add_numbers" is 3 tokens, the model picks token-by-token.
# After picking the first token (say id=14176 → "fn"), we must ensure
# the NEXT token continues a valid function name — not just any token.
# Filtering the encoded ids of every function name step by step solves this.

print()
print("Conclusion: function names may span multiple tokens.")
print("choose() (test_09) constrains each token so only valid names can form.")
