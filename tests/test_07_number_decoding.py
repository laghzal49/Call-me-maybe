"""
CONCEPT 7 — NUMBER DECODING (digit accumulation)
=================================================
JSON number grammar:   [-] digit+ [. digit+]

The model generates one token at a time, constrained to tokens that keep
the number valid.  We stop as soon as the model picks an END token.

Rules that change the allowed set at each step:
  ALWAYS         : digit_ids (one or more digits)
  AT START       : + minus_id (sign only at position 0)
  AFTER A DIGIT  : + end_ids (may stop — number is already valid)
  AFTER A DIGIT,
    if not integer: + dot_id (only once, never again after that)

The END TOKEN is NOT appended to self.ids.
The code (_emit) will write the separator (, or }) after the number.

Run: uv run python tests/test_07_number_decoding.py   (no model needed)
"""

# ── fake token sets (mirror Decoder.__init__) ─────────────────────────────────
DIGIT_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}   # one token per digit character
DOT_ID = 10
MINUS_ID = 11
END_IDS = {12, 13}    # comma=12, closing-brace=13

# id → text for decoding
ID_TO_TEXT = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: ".", 11: "-", 12: ",", 13: "}",
}


def allowed_set(text: str, has_digit: bool, has_dot: bool, integer_only: bool) -> set:
    """Build the allowed token set for the current position in a number."""
    allowed = set(DIGIT_IDS)
    if not text and MINUS_ID != -1:
        allowed.add(MINUS_ID)             # sign only at position 0
    if not integer_only and has_digit and not has_dot and DOT_ID != -1:
        allowed.add(DOT_ID)               # one dot, only after a digit
    if has_digit:
        allowed |= END_IDS                # may stop once we have at least one digit
    return allowed


def simulate_number(token_sequence: list, integer_only: bool = False) -> float:
    """
    Simulate gen_number() with a pre-scripted token sequence.
    Each entry: (token_id, is_end_token)
    """
    text = ""
    has_digit = has_dot = False

    for tok_id, is_end in token_sequence:
        current_allowed = allowed_set(text, has_digit, has_dot, integer_only)
        assert tok_id in current_allowed, (
            f"token {tok_id} not in allowed set {current_allowed}"
        )
        if is_end:
            print(f"  tok={tok_id} ({ID_TO_TEXT[tok_id]!r})  END → stop")
            break
        piece = ID_TO_TEXT[tok_id]
        text += piece
        has_digit = has_digit or any(c.isdigit() for c in piece)
        has_dot = has_dot or piece == "."
        print(f"  tok={tok_id} ({piece!r})  text so far: {text!r}")

    value = (int(text) if integer_only else float(text)) if text else 0.0
    return value


# ── test 1: float "265.5" ─────────────────────────────────────────────────────
print("Test 1: decoding float 265.5")
tokens_float = [
    (2, False),   # "2"
    (6, False),   # "6"
    (5, False),   # "5"
    (10, False),  # "."
    (5, False),   # "5"
    (12, True),   # "," → END, not appended
]
result = simulate_number(tokens_float, integer_only=False)
assert result == 265.5
print(f"  result: {result}")

# ── test 2: integer "42" ─────────────────────────────────────────────────────
print("\nTest 2: decoding integer 42")
tokens_int = [
    (4, False),   # "4"
    (2, False),   # "2"
    (13, True),   # "}" → END
]
result_int = simulate_number(tokens_int, integer_only=True)
assert result_int == 42
print(f"  result: {result_int}")

# ── test 3: negative float "-3.14" ───────────────────────────────────────────
print("\nTest 3: decoding -3.14")
tokens_neg = [
    (11, False),  # "-"
    (3, False),   # "3"
    (10, False),  # "."
    (1, False),   # "1"
    (4, False),   # "4"
    (12, True),   # ","  → END
]
result_neg = simulate_number(tokens_neg, integer_only=False)
assert result_neg == -3.14
print(f"  result: {result_neg}")

# ── test 4: dot not allowed for integer type ──────────────────────────────────
print("\nTest 4: dot is NOT in allowed set for integer_only=True")
step_allowed = allowed_set("3", has_digit=True, has_dot=False, integer_only=True)
assert DOT_ID not in step_allowed
print(f"  allowed after '3' (integer): {step_allowed}")
print("  dot_id NOT in set — dot is forbidden for integer params")

print("\nOK: number decoding handles float, integer, negative correctly")
