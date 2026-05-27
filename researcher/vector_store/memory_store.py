import numpy as np

from researcher.embeddings.base import BaseEmbedder


class MemoryVectorStore:
    """In-memory vector store with cosine similarity search.

    All vectors are L2-normalised on insertion so that dot-product scores
    equal cosine similarity at query time — no external database required.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.vectors: np.ndarray | None = None  # shape: (n_chunks, dimensions)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    async def add_documents(self, documents: list[str], metadatas: list[dict]) -> None:
        """Embed *documents*, normalise, and append to the store.

        Args:
            documents: Raw text chunks to store.
            metadatas: One dict per chunk, e.g. ``{"url": str, "chunk_index": int}``.
        """
        if not documents:
            return

        embeddings = await self.embedder.embed_documents(documents)
        new_vectors = np.array(embeddings, dtype=np.float32)

        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        new_vectors = new_vectors / norms

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    async def clear(self) -> None:
        """Reset the store — call between research runs if reusing the instance."""
        self.documents = []
        self.metadatas = []
        self.vectors = None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.42,
    ) -> list[dict]:
        """Return up to *k* chunks whose cosine similarity to *query* >= *threshold*.

        Args:
            query:     The search string to embed and compare against.
            k:         Maximum number of results to return.
            threshold: Minimum cosine similarity score to include (0–1).

        Returns:
            List of ``{"content": str, "metadata": dict, "score": float}``
            sorted by score descending.
        """
        if self.vectors is None or len(self.documents) == 0:
            return []

        query_vec = np.array(
            await self.embedder.embed_query(query), dtype=np.float32
        )
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = np.dot(self.vectors, query_vec)  # (n_chunks,)

        above_threshold = np.where(scores >= threshold)[0]
        if len(above_threshold) == 0:
            return []

        sorted_indices = above_threshold[np.argsort(scores[above_threshold])[::-1]]
        top_indices = sorted_indices[:k]

        return [
            {
                "content": self.documents[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a summary of what is currently stored."""
        return {
            "total_chunks": len(self.documents),
            "vector_dimensions": self.vectors.shape[1] if self.vectors is not None else None,
        }
