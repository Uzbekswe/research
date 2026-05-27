import logging
import os

from .base import BaseRetriever

logger = logging.getLogger(__name__)


class TavilySearch(BaseRetriever):
    """Search retriever backed by the Tavily Search API.

    Requires ``TAVILY_API_KEY`` in the environment.  Uses ``search_depth="advanced"``
    to get richer, more relevant snippets suitable for RAG.

    Tavily result fields:
      ``url``     → mapped to ``"href"``
      ``title``   → ``"title"``
      ``content`` → ``"body"``
    """

    def __init__(self, query: str) -> None:
        super().__init__(query)
        try:
            from tavily import TavilyClient  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'tavily-python' package is required for TavilySearch. "
                "Install it with: pip install -U tavily-python"
            ) from exc

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY environment variable is not set."
            )
        self._client = TavilyClient(api_key=api_key)

    def search(self, max_results: int = 5) -> list[dict]:
        """Run a Tavily web search with advanced depth.

        Args:
            max_results: Maximum number of results (default 5).

        Returns:
            List of ``{"href", "title", "body"}`` dicts.
        """
        try:
            response = self._client.search(
                query=self.query,
                search_depth="advanced",
                max_results=max_results,
            )
            raw: list[dict] = response.get("results", [])
        except Exception as exc:
            logger.warning("TavilySearch failed for %r: %s", self.query, exc)
            raw = []

        return [
            {
                "href": r.get("url", ""),
                "title": r.get("title", ""),
                "body": r.get("content", ""),
            }
            for r in raw
        ]
