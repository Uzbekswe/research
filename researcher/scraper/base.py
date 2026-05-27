from abc import ABC, abstractmethod

import httpx


class BaseScraper(ABC):
    """Common interface for all page scrapers.

    Each concrete scraper receives a shared async HTTP session at construction
    time so that connection pools and headers are reused across many scrapes.
    """

    def __init__(self, url: str, session: httpx.AsyncClient) -> None:
        self.url = url
        self.session = session

    @abstractmethod
    async def scrape(self) -> tuple[str, str]:
        """Fetch and clean the page at ``self.url``.

        Returns:
            A 2-tuple of:
              - ``raw_content`` – cleaned plain text extracted from the page.
              - ``image_urls``  – JSON array string of ``<img>`` src URLs found
                on the page, e.g. ``'["https://example.com/a.jpg"]'`` or
                ``'[]'`` when none are found.

        Implementations must never raise; exceptions should be caught
        internally and an empty string pair returned instead.
        """
