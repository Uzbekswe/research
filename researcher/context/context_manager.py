"""Research context assembly — embedding-based RAG retrieval.

Replaces the Layer 1 length-based concatenation with proper similarity
search: content is chunked, embedded, stored in a MemoryVectorStore, then
retrieved by cosine similarity against the research query and all sub-queries.
"""

import logging

from researcher.config import Config
from researcher.embeddings import get_embedder
from researcher.vector_store import MemoryVectorStore

from .chunker import chunk_documents

logger = logging.getLogger(__name__)


async def build_context_store(sources: list[dict], cfg: Config) -> MemoryVectorStore:
    """Chunk and embed all scraped sources into an in-memory vector store.

    Args:
        sources: Source dicts from ResearchMemory with keys
                 ``"url"``, ``"content"``, ``"summary"``.
        cfg:     Researcher config supplying EMBEDDING, CHUNK_SIZE, CHUNK_OVERLAP.

    Returns:
        Populated :class:`MemoryVectorStore` ready for similarity search.
    """
    embedder = get_embedder(cfg.EMBEDDING)
    store = MemoryVectorStore(embedder)

    if not sources:
        return store

    # ResearchMemory stores content under "content"; chunk_documents expects
    # "raw_content" — normalise here so the chunker stays generic.
    docs_for_chunking = [
        {"url": s.get("url", ""), "raw_content": s.get("content", "")}
        for s in sources
    ]

    chunks, metadatas = chunk_documents(
        docs_for_chunking,
        chunk_size=cfg.CHUNK_SIZE,
        overlap=cfg.CHUNK_OVERLAP,
    )

    await store.add_documents(chunks, metadatas)

    logger.info(
        "📦 Built vector store: %d chunks from %d sources",
        store.get_stats()["total_chunks"],
        len(sources),
    )
    return store


async def get_research_context(
    query: str,
    sub_queries: list[str],
    store: MemoryVectorStore,
    cfg: Config,
) -> str:
    """Retrieve the most relevant chunks for the research query.

    Runs similarity search for the main *query* and every *sub_query*,
    deduplicates results by content, sorts by score, and formats the top
    ``cfg.MAX_CONTEXT_CHUNKS`` into a context string.

    Args:
        query:       The primary research question.
        sub_queries: Additional sub-queries generated during research.
        store:       Populated vector store from :func:`build_context_store`.
        cfg:         Config supplying SIMILARITY_THRESHOLD, MAX_CONTEXT_CHUNKS.

    Returns:
        Assembled context string with attributed ``Source:`` blocks, or ``""``
        if the store is empty or no chunks pass the similarity threshold.
    """
    if store.get_stats()["total_chunks"] == 0:
        return ""

    all_queries = [query] + sub_queries

    # Keyed by content to deduplicate identical chunks retrieved via multiple queries.
    seen: dict[str, dict] = {}
    total_retrieved = 0

    for q in all_queries:
        results = await store.similarity_search(
            query=q,
            k=10,
            threshold=cfg.SIMILARITY_THRESHOLD,
        )
        total_retrieved += len(results)
        for r in results:
            content = r["content"]
            if content not in seen or r["score"] > seen[content]["score"]:
                seen[content] = r

    unique_chunks = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
    top_chunks = unique_chunks[: cfg.MAX_CONTEXT_CHUNKS]

    unique_urls = {c["metadata"].get("url", "") for c in top_chunks}

    logger.info(
        "🎯 Context: %d unique chunks retrieved (from %d total, threshold=%.2f)",
        len(top_chunks),
        total_retrieved,
        cfg.SIMILARITY_THRESHOLD,
    )
    logger.info("   Covering %d unique sources", len(unique_urls))

    blocks: list[str] = []
    for chunk in top_chunks:
        url = chunk["metadata"].get("url", "unknown")
        blocks.append(f"Source: {url}\n{chunk['content']}\n\n---\n\n")

    return "".join(blocks)
