from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one float vector per text."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string, returning one float vector."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector size produced by this embedder."""
