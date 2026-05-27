"""Layer 3 RAG unit tests — no real API calls.

All embedding calls use a MockEmbedder with deterministic fixed vectors so
tests are fast, hermetic, and reproducible without any external services.
"""

import numpy as np
import pytest

from researcher.context.chunker import chunk_text
from researcher.embeddings.base import BaseEmbedder
from researcher.vector_store import MemoryVectorStore


# ---------------------------------------------------------------------------
# Shared mock embedder
# ---------------------------------------------------------------------------

class MockEmbedder(BaseEmbedder):
    """Returns fixed 3-D vectors for a known vocabulary; falls back to zeros.

    Document vectors (pre-normalisation):
      "cats are animals"  → [1.0, 0.0, 0.0]   cosine with query = 1.000
      "dogs are pets"     → [0.7, 0.7, 0.0]   cosine with query ≈ 0.707
      "python programming"→ [0.0, 0.0, 1.0]   cosine with query = 0.000

    Query "cats" → [1.0, 0.0, 0.0] (already unit length)

    Note: [0.9, 0.1, 0] would normalise to cosine ≈ 0.993 against [1,0,0],
    which would exceed a 0.99 threshold — [0.7, 0.7, 0] gives a clear 0.707
    so threshold filtering in test_threshold_filtering() works as intended.
    """

    _VECTORS: dict[str, list[float]] = {
        "cats are animals":   [1.0, 0.0, 0.0],
        "dogs are pets":      [0.7, 0.7, 0.0],
        "python programming": [0.0, 0.0, 1.0],
    }
    _QUERY_VECTORS: dict[str, list[float]] = {
        "cats": [1.0, 0.0, 0.0],
    }

    @property
    def dimensions(self) -> int:
        return 3

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._VECTORS.get(t, [0.0, 0.0, 0.0]) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._QUERY_VECTORS.get(text, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Helper: build a populated store from the shared vocabulary
# ---------------------------------------------------------------------------

async def _make_store() -> MemoryVectorStore:
    store = MemoryVectorStore(MockEmbedder())
    await store.add_documents(
        ["cats are animals", "dogs are pets", "python programming"],
        [{"url": "a"}, {"url": "b"}, {"url": "c"}],
    )
    return store


# ---------------------------------------------------------------------------
# Test 1 — chunker: sliding window produces multiple chunks
# ---------------------------------------------------------------------------

def test_chunker_basic():
    text = "word " * 600          # 600 words
    chunks = chunk_text(text, chunk_size=200, overlap=50)

    assert len(chunks) > 1, "Expected multiple chunks for 600-word text"
    # Each chunk should be at most chunk_size words (allow tiny rounding overage)
    assert all(len(c.split()) <= 210 for c in chunks), (
        f"Chunk too long: {max(len(c.split()) for c in chunks)} words"
    )


# ---------------------------------------------------------------------------
# Test 2 — chunker: short text returns single chunk unchanged
# ---------------------------------------------------------------------------

def test_chunker_short_text():
    text = "This is a short sentence."
    chunks = chunk_text(text)
    assert chunks == [text]


# ---------------------------------------------------------------------------
# Test 3 — MemoryVectorStore: top result matches most similar document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_store_similarity():
    store = await _make_store()

    results = await store.similarity_search("cats", k=2, threshold=0.0)

    assert len(results) == 2
    assert results[0]["content"] == "cats are animals", (
        f"Expected 'cats are animals' as top result, got {results[0]['content']!r}"
    )
    assert results[0]["score"] > results[1]["score"], (
        f"Top score {results[0]['score']:.4f} should exceed second {results[1]['score']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — MemoryVectorStore: threshold filters low-similarity chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_threshold_filtering():
    store = await _make_store()

    # "dogs are pets" has cosine ≈ 0.707 and "python programming" has 0.0,
    # so only "cats are animals" (cosine = 1.0) clears the 0.99 threshold.
    results = await store.similarity_search("cats", k=10, threshold=0.99)

    assert len(results) == 1, (
        f"Expected 1 result above threshold 0.99, got {len(results)}: "
        + str([(r['content'], round(r['score'], 4)) for r in results])
    )
    assert results[0]["content"] == "cats are animals"


# ---------------------------------------------------------------------------
# Test 5 — MemoryVectorStore: vectors are unit-normalised on insertion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vectors_are_unit_normalised():
    store = await _make_store()
    norms = np.linalg.norm(store.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"Not unit-normalised: {norms}"


# ---------------------------------------------------------------------------
# Test 6 — MemoryVectorStore: k cap is respected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_k_cap():
    store = await _make_store()
    results = await store.similarity_search("cats", k=1, threshold=0.0)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Test 7 — MemoryVectorStore: result dict has required keys
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_structure():
    store = await _make_store()
    results = await store.similarity_search("cats", k=1, threshold=0.0)

    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) == {"content", "metadata", "score"}
    assert isinstance(r["score"], float)
    assert isinstance(r["metadata"], dict)
    assert r["metadata"]["url"] == "a"


# ---------------------------------------------------------------------------
# Test 8 — MemoryVectorStore: empty store returns []
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_store_returns_empty():
    store = MemoryVectorStore(MockEmbedder())
    results = await store.similarity_search("cats", k=10, threshold=0.0)
    assert results == []


# ---------------------------------------------------------------------------
# Test 9 — DeepResearcher: new instance attributes exist
# ---------------------------------------------------------------------------

def test_researcher_new_attributes():
    from researcher import DeepResearcher
    r = DeepResearcher(query="test")
    assert r.sub_queries == []
    assert r.vector_store is None
    assert r.get_sub_queries() == []
    assert r.get_vector_store() is None
