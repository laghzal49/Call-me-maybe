"""
CONCEPT 5 — TOKEN SETS (which token ids are valid per type)
============================================================
To constrain different JSON value types, we classify every token id
in the vocabulary into sets that are valid in each context:

  digit_ids   — tokens whose text is ENTIRELY digits: "0","1",…,"42",…
  dot_id      — the single token for "."
  minus_id    — the single token for "-"
  end_ids     — first token of each end character: "," "}" "]" " " "\\n"
  quote_ids   — tokens whose text CONTAINS a double-quote: '"', '",', …

These sets are built ONCE in Decoder.__init__() and reused every step.
The "allowed" set passed to _pick() is assembled from them at each step.

Run: uv run python tests/test_05_token_sets.py   (no model needed)
"""

# ── 1. build sets from a fake vocabulary ─────────────────────────────────────
# The real vocab comes from llm.get_path_to_vocab_file() — a JSON file
# mapping token text → token id.  We replicate it here with small numbers.

fake_vocab = {
    "0": 0,  "1": 1,  "2": 2,  "3": 3,  "4": 4,
    "5": 5,  "6": 6,  "7": 7,  "8": 8,  "9": 9,
    "42": 10,  "100": 11,      # multi-digit tokens are still all-digits
    ".": 12,
    "-": 13,
    ",": 14,  "}": 15,  " ": 16,
    '"': 17,  '",': 18,  '"}': 19,  'world"': 20,  # all contain "
    "hello": 21,  "world": 22,
}

digit_ids = {i for t, i in fake_vocab.items() if t and all(c.isdigit() for c in t)}
dot_id = fake_vocab.get(".", -1)
minus_id = fake_vocab.get("-", -1)
quote_ids = {i for t, i in fake_vocab.items() if '"' in t}
# end_ids: in real code we call encode_ids(llm, c)[0] for each character.
# Here we look them up directly.
end_ids = {fake_vocab[c] for c in (",", "}", " ") if c in fake_vocab}

assert digit_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
assert dot_id == 12
assert minus_id == 13
assert quote_ids == {17, 18, 19, 20}   # every token that CONTAINS "
assert end_ids == {14, 15, 16}

print("Token sets built from vocabulary:")
print("  digit_ids :", sorted(digit_ids))
print("  dot_id    :", dot_id)
print("  minus_id  :", minus_id)
print("  quote_ids :", sorted(quote_ids))
print("  end_ids   :", sorted(end_ids))

# ── 2. how the allowed set changes while decoding a number ───────────────────
# _number() assembles a different allowed set at each step.

print("\nAllowed set as a number is built up:")


def allowed_for_number(
    text: str,
    has_digit: bool,
    has_dot: bool,
    integer_only: bool = False,
) -> set:
    """Mirror the logic in Decoder._number()."""
    allowed = set(digit_ids)
    if not text and minus_id != -1:          # sign only at the very start
        allowed.add(minus_id)
    if not integer_only and has_digit and not has_dot and dot_id != -1:
        allowed.add(dot_id)                  # one dot, only after a digit
    if has_digit:
        allowed |= end_ids                   # may stop once a digit exists
    return allowed


step0 = allowed_for_number("",  False, False)
step1 = allowed_for_number("-", False, False)   # after "-": minus no longer allowed
step2 = allowed_for_number("3", True,  False)   # after "3": dot and end allowed
step3 = allowed_for_number("3.", True,  True)   # after "3.": no second dot

print("  start  :", "digits + minus" if minus_id in step0 else "digits only")
print("  after -:", "digits only    (minus gone)")
print("  after 3:", "digits + dot + end  (can continue or stop)")
print("  after 3.:", "digits + end   (no second dot)")

assert minus_id in step0     # sign allowed at start
assert minus_id not in step1  # sign NOT allowed after something is written
assert dot_id in step2        # dot allowed after a digit
assert dot_id not in step3    # dot NOT allowed after the dot was already used

print("\nOK: token sets correctly model JSON number grammar")

# ── 3. why quote_ids uses containment, not equality ──────────────────────────
# BPE can bundle text + closing quote into one token.
# For example token 'world"' (id=20) would close the string "world"
# while also providing the last characters.  If we only matched on t == '"'
# we would miss tokens like '",', '"}', 'world"' and the string would
# never close properly.

print("\nWhy containment for quote_ids:")
for tok_text, tok_id in fake_vocab.items():
    if '"' in tok_text:
        print(f"  id={tok_id:2d}  text={tok_text!r:10s}  → valid string closer")
