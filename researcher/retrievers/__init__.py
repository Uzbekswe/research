from .base import BaseRetriever
from .duckduckgo import DuckDuckGoSearch
from .serper import SerperSearch
from .tavily import TavilySearch

_REGISTRY: dict[str, type[BaseRetriever]] = {
    "duckduckgo": DuckDuckGoSearch,
    "tavily": TavilySearch,
    "serper": SerperSearch,
}


def get_retriever(retriever_name: str) -> type[BaseRetriever]:
    """Return the retriever *class* for the given name string.

    Args:
        retriever_name: One of ``"duckduckgo"``, ``"tavily"``, ``"serper"``.

    Returns:
        The retriever class (not an instance).  Instantiate it yourself with
        the search query: ``get_retriever("tavily")(query="my question")``.

    Raises:
        ValueError: If ``retriever_name`` is not a registered retriever.

    Example::

        RetrieverClass = get_retriever("duckduckgo")
        results = RetrieverClass(query="who is Nikola Tesla?").search(max_results=5)
    """
    cls = _REGISTRY.get(retriever_name.lower().strip())
    if cls is None:
        supported = ", ".join(f'"{k}"' for k in sorted(_REGISTRY))
        raise ValueError(
            f"Unknown retriever {retriever_name!r}. "
            f"Supported retrievers: {supported}"
        )
    return cls


__all__ = [
    "BaseRetriever",
    "DuckDuckGoSearch",
    "TavilySearch",
    "SerperSearch",
    "get_retriever",
]
