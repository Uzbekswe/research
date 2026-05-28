import os

from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider


class VesslProvider(BaseLLMProvider):
    """Thin wrapper around OpenAIProvider pointing at a VESSL vLLM endpoint.

    Required env vars:
      VESSL_BASE_URL  — e.g. https://model-service-gateway-xxx.vessl.ai/request/v1
      VESSL_API_KEY   — the authorization token shown in the VESSL "Request" dialog
    """

    def __init__(self, model: str, temperature: float):
        base_url = os.getenv("VESSL_BASE_URL", "").rstrip("/")
        if not base_url:
            raise EnvironmentError(
                "VESSL_BASE_URL is not set. Copy it from the VESSL service 'Request' dialog."
            )
        self._delegate = OpenAIProvider(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=os.getenv("VESSL_API_KEY", "token"),
        )

    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        result = await self._delegate.get_chat_response(messages, max_tokens)
        # OPUS FIX (3I): forward token usage so the unified cost path works.
        self.last_usage = self._delegate.last_usage
        return result
