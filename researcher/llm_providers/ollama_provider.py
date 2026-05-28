from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider

_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str, temperature: float):
        self._delegate = OpenAIProvider(
            model=model,
            temperature=temperature,
            base_url=_OLLAMA_BASE_URL,
            api_key="ollama",  # openai client requires a non-empty string
        )

    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        result = await self._delegate.get_chat_response(messages, max_tokens)
        # OPUS FIX (3I): forward token usage so the unified cost path works.
        self.last_usage = self._delegate.last_usage
        return result
