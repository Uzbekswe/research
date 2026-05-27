from researcher.config import Config

from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider
from .google_provider import GoogleProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .vessl_provider import VesslProvider

_PROVIDERS = {
    "openai": lambda model, temp, _max: OpenAIProvider(model=model, temperature=temp),
    "anthropic": lambda model, temp, _max: AnthropicProvider(model=model, temperature=temp),
    "google": lambda model, temp, _max: GoogleProvider(model=model, temperature=temp),
    "groq": lambda model, temp, _max: GroqProvider(model=model, temperature=temp),
    "ollama": lambda model, temp, _max: OllamaProvider(model=model, temperature=temp),
    "vessl": lambda model, temp, _max: VesslProvider(model=model, temperature=temp),
}


def get_llm_provider(llm_string: str, temperature: float, max_tokens: int) -> BaseLLMProvider:
    """Instantiate the correct LLM provider from a 'provider:model' string.

    Args:
        llm_string: e.g. "openai:gpt-4o-mini" or "anthropic:claude-haiku-4-5"
        temperature: sampling temperature
        max_tokens: default max tokens (passed through for providers that need it at init)

    Raises:
        ValueError: if the provider name is not recognised
    """
    provider_name, model = Config.parse_provider_and_model(llm_string)
    factory = _PROVIDERS.get(provider_name.lower())
    if factory is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown LLM provider {provider_name!r}. Supported providers: {supported}"
        )
    return factory(model, temperature, max_tokens)


__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "GroqProvider",
    "OllamaProvider",
    "VesslProvider",
    "get_llm_provider",
]
