import logging
import time

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# Minimum delay between consecutive DDGS calls to avoid triggering rate limits.
_RATE_LIMIT_SLEEP = 0.5  # seconds


class DuckDuckGoSearch(BaseRetriever):
    """Search retriever backed by DuckDuckGo via the ``ddgs`` package.

    No API key is required.  A small sleep is inserted after each call to
    reduce the likelihood of being rate-limited by DuckDuckGo.

    The ``ddgs`` DDGS.text() method already returns dicts with the keys
    ``"href"``, ``"title"``, and ``"body"``, so no field mapping is needed.
    """

    def __init__(self, query: str) -> None:
        super().__init__(query)
        try:
            from ddgs import DDGS  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'ddgs' package is required for DuckDuckGoSearch. "
                "Install it with: pip install -U ddgs"
            ) from exc
        self._ddgs = DDGS()

    def search(self, max_results: int = 5) -> list[dict]:
        """Run a DuckDuckGo web text search.

        Args:
            max_results: Maximum number of results (default 5).

        Returns:
            List of ``{"href", "title", "body"}`` dicts.
        """
        try:
            results: list[dict] = self._ddgs.text(
                self.query,
                region="wt-wt",
                safesearch="moderate",
                max_results=max_results,
            )
        except Exception as exc:
            logger.warning("DuckDuckGoSearch failed for %r: %s", self.query, exc)
            results = []

        time.sleep(_RATE_LIMIT_SLEEP)
        return results or []
