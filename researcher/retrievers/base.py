from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Common interface for all search retrievers.

    Every concrete retriever accepts the search query at construction time and
    exposes a single ``search`` method.  Results always use the same dict shape
    so the rest of the pipeline can treat all retrievers interchangeably.
    """

    def __init__(self, query: str) -> None:
        self.query = query

    @abstractmethod
    def search(self, max_results: int = 5) -> list[dict]:
        """Execute the search and return a normalised result list.

        Args:
            max_results: Maximum number of results to return.

        Returns:
            List of dicts, each containing:
              - ``"href"``  – canonical URL of the result page
              - ``"title"`` – page title
              - ``"body"``  – short text snippet / description
        """
