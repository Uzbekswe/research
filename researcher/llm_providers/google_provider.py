import logging
import os

from google import genai
from google.genai import types

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseLLMProvider):
    def __init__(self, model: str, temperature: float):
        self.model_name = model
        self.temperature = temperature
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    @staticmethod
    def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[types.Content]]:
        """Split system instruction from user/assistant turns.

        Returns (system_instruction, contents).
        Gemini uses "user"/"model" roles; system is a top-level instruction.
        """
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=text)]))
            else:
                contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

        return "\n\n".join(system_parts), contents

    async def get_chat_response(self, messages: list[dict], max_tokens: int) -> str:
        # OPUS FIX: reset usage at start so a failed call leaves clean state.
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        system_instruction, contents = self._to_gemini_contents(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction or None,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        usage = response.usage_metadata
        if usage:
            logger.info(
                "google usage — prompt: %d, candidates: %d, total: %d",
                usage.prompt_token_count,
                usage.candidates_token_count,
                usage.total_token_count,
            )
            # OPUS FIX (3I): expose tokens for cost tracking.
            self.last_usage = {
                "prompt_tokens": usage.prompt_token_count or 0,
                "completion_tokens": usage.candidates_token_count or 0,
            }

        # OPUS FIX: response.text may be None when the model returns no candidates;
        # the public signature promises a str so coerce.
        return response.text or ""
