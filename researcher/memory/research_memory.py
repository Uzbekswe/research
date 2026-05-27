"""In-process state container for a single research run.

ResearchMemory is intentionally simple: it holds exactly what one research
session needs to track — visited URLs, accumulated sources, images, and a
running cost tally.  It has no persistence; a new instance is created for
every research run.

Modelled after gpt_researcher/memory/research.py.
"""

import logging

logger = logging.getLogger(__name__)


class ResearchMemory:
    """Mutable state bag for one end-to-end research session.

    All mutation goes through the public methods so that deduplication logic
    and any future instrumentation live in one place.

    Attributes:
        visited_urls:   Set of URLs that have already been scraped this run.
                        Used to skip duplicate fetches across multiple search
                        iterations.
        research_costs: Accumulated LLM token-cost total (float, in USD or
                        arbitrary units — matches whatever the cost_callback
                        produces).
        images:         Flat list of image URLs discovered across all scraped
                        pages.
        sources:        Ordered list of source dicts added via :meth:`add_source`.
                        Each entry has the shape
                        ``{"url": str, "content": str, "summary": str}``.
    """

    def __init__(self) -> None:
        self.visited_urls: set[str] = set()
        self.research_costs: float = 0.0
        self.images: list[str] = []
        self.sources: list[dict] = []

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def add_source(self, url: str, content: str, summary: str = "") -> None:
        """Add a scraped source to memory, deduplicating by URL.

        If *url* has already been visited this run the call is a no-op, so
        callers do not need to guard against duplicates themselves.

        Args:
            url:     Canonical URL of the scraped page.
            content: Raw (cleaned) page text from the scraper.
            summary: Optional query-relevant summary produced by FAST_LLM.
                     Defaults to ``""`` when summarisation is skipped.
        """
        if self.is_visited(url):
            logger.debug("Skipping duplicate URL: %s", url)
            return

        self.visited_urls.add(url)
        self.sources.append({"url": url, "content": content, "summary": summary})
        logger.debug("Added source %s (%d content chars)", url, len(content))

    def add_source_no_visit_check(self, url: str, content: str, summary: str = "") -> None:
        """Add a source that has already been atomically claimed via visited_urls.

        Use this only when the caller has already done:
            if url not in self.visited_urls:
                self.visited_urls.add(url)   # atomic claim
                ...
                self.add_source_no_visit_check(url, content, summary)

        This skips the duplicate guard so the URL is not double-counted.
        """
        self.sources.append({"url": url, "content": content, "summary": summary})
        logger.debug("Added source %s (%d content chars)", url, len(content))

    def is_visited(self, url: str) -> bool:
        """Return True if *url* has already been scraped this run."""
        return url in self.visited_urls

    def get_context(self, max_sources: int = 20) -> list[dict]:
        """Return accumulated sources, capped at *max_sources*.

        Sources are returned in insertion order (i.e. the order they were
        scraped).  Callers that need relevance-ranked context should use
        :mod:`researcher.context.context_manager` instead.

        Args:
            max_sources: Maximum number of source dicts to return.

        Returns:
            A slice of :attr:`sources` of length ≤ *max_sources*.
        """
        return self.sources[:max_sources]

    def get_source_urls(self) -> list[str]:
        """Return an ordered list of all visited URLs.

        The order matches insertion order of :attr:`visited_urls` as of
        Python 3.7+ dict semantics applied to sets (actually sets are
        unordered — we derive order from :attr:`sources` to preserve
        the scrape sequence).
        """
        return [s["url"] for s in self.sources]

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def add_costs(self, cost: float) -> None:
        """Accumulate *cost* into the running total.

        Args:
            cost: Additional cost to add (same unit as whatever the LLM
                  provider reports; typically USD or token count).
        """
        self.research_costs += cost

    def get_costs(self) -> float:
        """Return the total accumulated cost for this research run."""
        return self.research_costs

    # ------------------------------------------------------------------
    # Image tracking
    # ------------------------------------------------------------------

    def add_images(self, image_urls: list[str]) -> None:
        """Extend the image list with newly discovered URLs (no dedup)."""
        self.images.extend(image_urls)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ResearchMemory("
            f"sources={len(self.sources)}, "
            f"visited={len(self.visited_urls)}, "
            f"images={len(self.images)}, "
            f"costs={self.research_costs:.4f})"
        )
