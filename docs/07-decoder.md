# 07 — `src/decoder.py`

## Role

The thin public entry point for generating one function call. It hides the state
machine behind a single function.

## Theory

A small "facade": callers should not need to know about the `StateMachine` class,
its phases, or how it is constructed. They call one function with clear arguments
and get one result. This keeps `__main__` simple and lets the internals change
without touching callers.

## Inputs / Outputs

- **Input:** `llm`, `vocab`, `ctx` (`GenerationContext`), `prompt: str`.
- **Output:** `JsonObject` = `{"prompt", "name", "parameters"}`.

## How it works

```python
def generate_call(llm, vocab, ctx, prompt):
    return StateMachine(llm, vocab, ctx, prompt).run()
```

Construct a fresh `StateMachine` for the prompt and run it. A new instance per
prompt means each call starts from a clean sequence and state — no leftover data
from the previous prompt.

## Why this design

- **Why a wrapper at all?** It is the stable public API. If we later change how
  generation works (different machine, batching, etc.), `generate_call`'s
  signature stays the same and `__main__` is untouched.
- **Why one machine per prompt?** Isolation: state and the token sequence are
  per-prompt, so there is nothing to reset between prompts.

## How to reimplement

1. Import `StateMachine`.
2. Write one function that builds it and returns `.run()`.

## Edge cases

- None of its own; it forwards errors from the state machine (caught per-prompt in
  `__main__`).
