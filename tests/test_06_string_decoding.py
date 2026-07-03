"""
CONCEPT 6 — STRING DECODING (the quote race)
=============================================
A JSON string value sits between two double-quotes: "hello"

The code writes the OPENING and CLOSING quotes (_emit('"')).
The model only generates the content tokens — but under a constraint:
at every step we compare the best content token against the best closing
token.  When the closing token wins the logit comparison, we stop.

Algorithm (gen_string in decoder.py):
  loop:
    lg = model logits for current sequence
    close = _pick(lg, quote_ids)       # best token CONTAINING "
    close_val = lg[close]              # save BEFORE we mask
    lg[quote_ids] = -inf               # mask quotes out
    best = argmax(lg)                  # best NON-QUOTE token
    if close_val >= lg[best]:
        salvage prefix before " in closing token
        break
    append best, continue

BPE quirk: a closing token can bundle content + quote, e.g. 'world"'.
We salvage "world" (the part before the ") so no characters are lost.

Run: uv run python tests/test_06_string_decoding.py   (no model needed)
"""

import numpy as np

# ── fake vocabulary for this demo ────────────────────────────────────────────
# id → text
token_text = {
    0: "he",
    1: "llo",
    2: '"',       # plain closing quote
    3: '",',      # quote + comma (closing token)
    4: 'llo"',    # content + quote bundled by BPE
    5: "xyz",
}

quote_ids = {i for i, t in token_text.items() if '"' in t}   # {2, 3, 4}

# ── helper: the gen_string() logic in isolation ─────────────────────────────────


def decode_string(steps: list) -> str:
    """
    Simulate gen_string() with pre-scripted logit arrays.
    steps = list of np.array, one per generation step.
    Returns the decoded string content (without surrounding quotes).
    """
    text = ""
    for step_idx, lg in enumerate(steps):
        lg = lg.copy()   # don't mutate the original

        # Best closing token (one that CONTAINS a quote)
        close_masked = np.full(len(lg), -np.inf)
        close_masked[list(quote_ids)] = lg[list(quote_ids)]
        close = int(np.argmax(close_masked))
        close_val = float(lg[close])   # save BEFORE masking — critical!

        # Best content token (everything else)
        lg[list(quote_ids)] = -np.inf
        best = int(np.argmax(lg))

        if close_val >= float(lg[best]):
            # Closing token won — extract any content before the "
            tok = token_text[close]
            q = tok.find('"')
            prefix = tok[:q] if q > 0 else ""
            if prefix:
                text += prefix
                print(f"  step {step_idx}: quote wins  → salvage {prefix!r}, stop")
            else:
                print(f"  step {step_idx}: quote wins  → stop")
            break
        else:
            piece = token_text[best]
            text += piece
            print(f"  step {step_idx}: content wins → append {piece!r}")

    return text


# ── test 1: clean close (no bundled content) ─────────────────────────────────
print("Test 1: encoding 'he' + plain close")
print("  (opening quote already written by _emit)")
steps1 = [
    np.array([3.0, 0.1, 0.1, 0.2, 0.3, 0.1]),  # "he" wins
    np.array([0.1, 0.1, 3.5, 0.3, 0.2, 0.1]),  # token 2 ('"') wins
]
result1 = decode_string(steps1)
assert result1 == "he"
print(f"  decoded string value: {result1!r}")

# ── test 2: bundled BPE close ('llo"') ───────────────────────────────────────
print("\nTest 2: BPE bundled closing token ('llo\"')")
steps2 = [
    np.array([3.0, 0.5, 0.1, 0.1, 0.1, 0.1]),  # "he" wins
    np.array([0.1, 0.1, 0.1, 0.1, 4.5, 0.1]),  # token 4 ('llo"') wins
]
result2 = decode_string(steps2)
assert result2 == "hello"   # "he" + salvaged "llo"
print(f"  decoded string value: {result2!r}")

# ── test 3: why we MUST save close_val before masking ─────────────────────────
print("\nWhy save close_val before masking:")
print("  After lg[quote_ids] = -inf, lg[close] becomes -inf.")
print("  Comparing -inf >= lg[best] is always False → string never closes.")
print("  Saving close_val = lg[close] BEFORE the mask avoids this bug.")

print("\nOK: string decoding correctly handles plain and BPE closing tokens")
