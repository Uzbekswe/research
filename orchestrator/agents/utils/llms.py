import logging

logger = logging.getLogger(__name__)


async def call_model(
    prompt: str,
    model: str,
    response_format: str = "text",
    max_tokens: int = 4000,
    temperature: float = 0.4,
    system_prompt: str = "",
) -> str:
    """Unified LLM caller for all orchestrator agents.

    Args:
        prompt:          The user message.
        model:           Provider:model string, e.g. ``"openai:gpt-4o"``.
        response_format: ``"json"`` appends a JSON-only instruction to the
                         system prompt; ``"text"`` leaves it unchanged.
        max_tokens:      Maximum tokens to generate.
        temperature:     Sampling temperature.
        system_prompt:   Optional system role instruction.

    Returns:
        Model response as a plain string.

    Raises:
        Re-raises any provider exception after logging it.
    """
    from researcher.llm_providers import get_llm_provider

    provider = get_llm_provider(model, temperature, max_tokens)

    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if response_format == "json":
        json_instruction = (
            "\nYou MUST respond with valid JSON only. "
            "No explanation, no markdown fences."
        )
        if messages:
            messages[0]["content"] += json_instruction
        else:
            messages.append({"role": "system", "content": json_instruction})

    messages.append({"role": "user", "content": prompt})

    try:
        return await provider.get_chat_response(messages, max_tokens)
    except Exception as exc:
        logger.warning("call_model error (%s): %s", model, exc)
        print(f"⚠️ call_model error ({model}): {exc}")
        raise
