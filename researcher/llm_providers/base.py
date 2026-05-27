from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        """Send messages in OpenAI format and return the assistant reply as a string.

        Args:
            messages: [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
            max_tokens: maximum tokens to generate
        """
