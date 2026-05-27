import asyncio
import logging
import os

import openai
from openai import AsyncOpenAI

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str,
        temperature: float,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                usage = response.usage
                if usage:
                    logger.info(
                        "openai usage — prompt: %d, completion: %d, total: %d",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                    )
                return response.choices[0].message.content or ""

            except openai.RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning("rate limit hit, retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, _MAX_RETRIES)
                await asyncio.sleep(delay)
