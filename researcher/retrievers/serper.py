import logging
import os

import httpx

from .base import BaseRetriever

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"
_TIMEOUT = 10.0  # seconds


class SerperSearch(BaseRetriever):
    """Search retriever backed by the Serper Google Search API.

    Requires ``SERPER_API_KEY`` in the environment.  Posts to
    ``https://google.serper.dev/search`` and maps the ``organic`` results
    to the standard retriever format.

    Serper result fields:
      ``link``    → mapped to ``"href"``
      ``title``   → ``"title"``
      ``snippet`` → ``"body"``
    """

    def __init__(self, query: str) -> None:
        super().__init__(query)
        self._api_key = os.getenv("SERPER_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "SERPER_API_KEY environment variable is not set."
            )

    def search(self, max_results: int = 5) -> list[dict]:
        """Run a Google search via the Serper API.

        Args:
            max_results: Maximum number of organic results to return (default 5).

        Returns:
            List of ``{"href", "title", "body"}`` dicts.
        """
        payload = {"q": self.query, "num": max_results}
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                _SERPER_URL,
                json=payload,
                headers=headers,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("SerperSearch failed for %r: %s", self.query, exc)
            return []

        organic: list[dict] = data.get("organic", [])
        return [
            {
                "href": r.get("link", ""),
                "title": r.get("title", ""),
                "body": r.get("snippet", ""),
            }
            for r in organic[:max_results]
        ]
