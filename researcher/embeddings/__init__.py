from researcher.config import Config

from .base import BaseEmbedder
from .huggingface_embedder import HuggingFaceEmbedder
from .ollama_embedder import OllamaEmbedder
from .openai_embedder import OpenAIEmbedder

_EMBEDDERS: dict[str, type[BaseEmbedder]] = {
    "openai": OpenAIEmbedder,
    "huggingface": HuggingFaceEmbedder,
    "ollama": OllamaEmbedder,
}


def get_embedder(embedding_string: str) -> BaseEmbedder:
    """Instantiate the correct embedder from a 'provider:model' string.

    Args:
        embedding_string: e.g. "openai:text-embedding-3-small" or
                          "huggingface:all-MiniLM-L6-v2"

    Raises:
        ValueError: if the provider name is not recognised
    """
    provider, model = Config.parse_provider_and_model(embedding_string)
    cls = _EMBEDDERS.get(provider.lower())
    if cls is None:
        supported = ", ".join(sorted(_EMBEDDERS))
        raise ValueError(
            f"Unknown embedding provider {provider!r}. Supported: {supported}"
        )
    return cls(model=model)


__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
    "HuggingFaceEmbedder",
    "OllamaEmbedder",
    "get_embedder",
]
