import os

from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(BaseLLMProvider):
    def __init__(self, model: str, temperature: float):
        self._delegate = OpenAIProvider(
            model=model,
            temperature=temperature,
            base_url=_GROQ_BASE_URL,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        return await self._delegate.get_chat_response(messages, max_tokens)
