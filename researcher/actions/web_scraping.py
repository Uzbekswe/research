"""Web scraping actions.

Handles the full retrieve-and-scrape cycle: search → URL extraction → scraping.
"""

import logging

from researcher.config import Config
from researcher.retrievers import get_retriever
from researcher.scraper.scraper import scrape_urls

logger = logging.getLogger(__name__)


async def search_and_scrape(
    query: str,
    cfg: Config,
    websocket=None,
) -> list[dict]:
    """Search the web for *query* and scrape the resulting pages.

    Pipeline:
      1. Instantiate the configured retriever (``cfg.RETRIEVER``).
      2. Call ``.search()`` to get up to ``cfg.MAX_SEARCH_RESULTS_PER_QUERY``
         results.
      3. Extract the ``"href"`` URL from each result.
      4. Pass the URL list to :func:`~researcher.scraper.scraper.scrape_urls`
         which runs concurrent async scraping under the semaphore cap.

    Args:
        query:     The search query string.
        cfg:       Researcher configuration.
        websocket: Optional WebSocket for progress streaming (reserved for
                   future use — not yet wired to the scraper layer).

    Returns:
        List of dicts from :func:`~researcher.scraper.scraper.scrape_urls`::

            [{"url": str, "raw_content": str, "image_urls": str}, ...]

        Empty list if the retriever or scraper step fails entirely.
    """
    # Support comma-separated hybrid retrieval: use the first provider.
    primary_retriever_name = cfg.RETRIEVER.split(",")[0].strip()

    try:
        RetrieverClass = get_retriever(primary_retriever_name)
        retriever = RetrieverClass(query=query)
        search_results = retriever.search(max_results=cfg.MAX_SEARCH_RESULTS_PER_QUERY)
    except Exception as exc:
        logger.error("Retriever %r failed for query %r: %s", primary_retriever_name, query, exc)
        return []

    urls = [r["href"] for r in search_results if r.get("href")]
    if not urls:
        logger.warning("No URLs returned by retriever for query %r", query)
        return []

    logger.info("Scraping %d URLs for query %r", len(urls), query)
    try:
        scraped = await scrape_urls(urls, cfg)
    except Exception as exc:
        logger.error("scrape_urls failed for query %r: %s", query, exc)
        return []

    return scraped


async def browse_web_sources(
    query: str,
    urls: list[str],
    cfg: Config,
    websocket=None,
) -> list[dict]:
    """Scrape a caller-supplied list of URLs directly (no search step).

    Used when the user has provided explicit source URLs rather than letting
    the retriever discover them.  The query is kept for logging and is passed
    through to the scraper config context.

    Args:
        query:     Research question (used for logging).
        urls:      Explicit list of URLs to scrape.
        cfg:       Researcher configuration.
        websocket: Optional WebSocket for progress streaming (reserved).

    Returns:
        Same format as :func:`search_and_scrape`.
    """
    if not urls:
        return []

    logger.info("Browsing %d explicit URLs for query %r", len(urls), query)
    try:
        return await scrape_urls(urls, cfg)
    except Exception as exc:
        logger.error("browse_web_sources failed: %s", exc)
        return []
