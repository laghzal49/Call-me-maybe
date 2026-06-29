"""Public entry for generating one function call from one prompt."""

from llm_sdk import Small_LLM_Model

from src.context import GenerationContext
from src.state_machine import JsonObject, StateMachine
from src.vocab import Vocab


def generate_call(
    llm: Small_LLM_Model,
    vocab: Vocab,
    ctx: GenerationContext,
    prompt: str,
) -> JsonObject:
    """Run the state machine for one prompt and return its function call."""
    return StateMachine(llm, vocab, ctx, prompt).run()
