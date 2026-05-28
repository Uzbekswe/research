# OPUS FIX (3I): centralised per-model pricing for cost tracking.
#
# Cost is computed as:
#   (prompt_tokens / 1e6) * input_price + (completion_tokens / 1e6) * output_price
#
# Prices are USD per 1M tokens.  Unknown models fall back to (0.0, 0.0) so
# cost tracking degrades to zero rather than crashing.

_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":          (2.50, 10.00),
    "gpt-4o-mini":     (0.15,  0.60),
    "gpt-4-turbo":     (10.00, 30.00),
    "gpt-3.5-turbo":   (0.50,  1.50),
    # Anthropic
    "claude-opus-4-5":     (15.00, 75.00),
    "claude-sonnet-4-5":   (3.00,  15.00),
    "claude-haiku-4-5":    (1.00,  5.00),
    # Google
    "gemini-1.5-pro":   (1.25,  5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    # Embeddings (input only, completion price treated as 0)
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost estimate for a single LLM call.

    Lookup is case-insensitive and matches by substring so that
    ``"gpt-4o-mini-2024-07-18"`` still resolves to the ``"gpt-4o-mini"`` entry.
    """
    if not model:
        return 0.0
    model_l = model.lower()
    for known, (in_price, out_price) in _PRICING_PER_MILLION.items():
        if known in model_l:
            return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
    return 0.0
