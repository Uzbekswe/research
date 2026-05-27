import logging

logger = logging.getLogger(__name__)

_MIN_CHUNK_WORDS = 20


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 100,
) -> list[str]:
    """Split *text* into overlapping word-count chunks.

    Args:
        text:       Raw document text.
        chunk_size: Target chunk size in words.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of non-empty chunk strings.  Returns ``[]`` for blank input and
        ``[text]`` when the text is shorter than *chunk_size* words.
    """
    stripped = text.strip()
    if not stripped:
        return []

    words = stripped.split()

    if len(words) <= chunk_size:
        return [stripped]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) >= _MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    overlap: int = 100,
) -> tuple[list[str], list[dict]]:
    """Chunk a list of source dicts into embedding-ready text pieces.

    Args:
        documents:  List of ``{"url": str, "raw_content": str}`` dicts.
        chunk_size: Passed through to :func:`chunk_text`.
        overlap:    Passed through to :func:`chunk_text`.

    Returns:
        A ``(chunks, metadatas)`` tuple where each metadata dict has the shape
        ``{"url": str, "chunk_index": int, "total_chunks": int}``.
    """
    chunks: list[str] = []
    metadatas: list[dict] = []

    for doc in documents:
        url = doc.get("url", "")
        text_chunks = chunk_text(doc.get("raw_content", ""), chunk_size, overlap)
        total = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadatas.append({"url": url, "chunk_index": i, "total_chunks": total})

    n_docs = len(documents)
    n_chunks = len(chunks)
    if n_docs:
        logger.debug(
            "Chunked %d documents into %d chunks (avg %.1f chunks/doc)",
            n_docs,
            n_chunks,
            n_chunks / n_docs,
        )

    return chunks, metadatas
