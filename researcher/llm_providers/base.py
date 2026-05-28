from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    # OPUS FIX (3I): subclasses set last_usage to {"prompt_tokens": int, "completion_tokens": int}
    # after every call so the agent layer can compute USD cost. Defaults to zeros so a
    # provider that fails to populate it produces 0.0 cost instead of crashing.
    last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}

    @abstractmethod
    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        """Send messages in OpenAI format and return the assistant reply as a string.

        Args:
            messages: [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
            max_tokens: maximum tokens to generate

        Implementations MUST populate ``self.last_usage`` with prompt/completion
        token counts after every successful call so the caller can attribute cost.
        """
