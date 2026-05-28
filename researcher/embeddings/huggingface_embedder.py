import asyncio

from .base import BaseEmbedder

_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        # OPUS FIX: asyncio.get_event_loop() is deprecated in 3.10+ and raises in
        # 3.12 when there is no running loop. asyncio.to_thread is the supported
        # equivalent and works with run_in_executor's default thread pool.
        result = await asyncio.to_thread(
            model.encode, texts, convert_to_numpy=True, show_progress_bar=False
        )
        return result.tolist()

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_documents([text])
        return results[0]

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS.get(self.model_name, 384)
