import asyncio

import httpx

from .base import BaseEmbedder

_OLLAMA_URL = "http://localhost:11434/api/embeddings"
_SEMAPHORE_LIMIT = 5

_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 1024,
}


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "nomic-embed-text") -> None:
        self.model = model

    async def embed_query(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _OLLAMA_URL,
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)

        async def _embed_one(text: str) -> list[float]:
            async with semaphore:
                return await self.embed_query(text)

        return list(await asyncio.gather(*(_embed_one(t) for t in texts)))

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS.get(self.model, 4096)
