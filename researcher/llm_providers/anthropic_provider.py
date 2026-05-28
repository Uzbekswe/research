import asyncio
import logging
import os

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """Extract leading system message; return (system_text, remaining_messages)."""
        system = ""
        rest = []
        for msg in messages:
            if msg["role"] == "system" and not rest:
                system = msg["content"]
            else:
                rest.append({"role": msg["role"], "content": msg["content"]})
        return system, rest

    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        # OPUS FIX: reset usage at start so a failed call leaves clean state.
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        system_text, anthropic_messages = self._split_messages(messages)

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    messages=anthropic_messages,
                )
                if system_text:
                    kwargs["system"] = system_text

                response = await self.client.messages.create(**kwargs)

                usage = response.usage
                logger.info(
                    "anthropic usage — input: %d, output: %d",
                    usage.input_tokens,
                    usage.output_tokens,
                )
                # OPUS FIX (3I): expose tokens via unified key names so the agent
                # cost callback works the same for every provider.
                self.last_usage = {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                }
                # OPUS FIX: guard against empty content list (rare API edge case).
                if not response.content:
                    return ""
                return response.content[0].text or ""

            except anthropic.RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning("rate limit hit, retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, _MAX_RETRIES)
                await asyncio.sleep(delay)
        # OPUS FIX: explicit fallthrough — should never happen, but reassures type checker.
        raise RuntimeError("Anthropic retry loop exited without returning a value")
