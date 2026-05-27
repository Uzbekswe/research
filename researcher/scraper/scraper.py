"""Main entry point for the scraping layer.

Call ``scrape_urls()`` with a list of URLs and a ``Config`` instance.
It manages the shared HTTP session, concurrency cap, and result aggregation.
"""

import asyncio
import logging

import httpx

from researcher.config import Config

from . import get_scraper

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 4.0  # seconds — matches GPT Researcher default


async def _scrape_one(
    url: str,
    session: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cfg: Config,
) -> dict | None:
    """Scrape a single URL under the semaphore and return a result dict.

    Returns ``None`` if the page yields no usable content (empty text, very
    short content, or any unhandled error).
    """
    ScraperClass = get_scraper(cfg.SCRAPER)
    async with sem:
        scraper = ScraperClass(url=url, session=session)
        try:
            raw_content, image_urls = await scraper.scrape()
        except Exception as exc:  # noqa: BLE001
            logger.error("Scrape failed for %s: %s", url, exc)
            return None

    if not raw_content or len(raw_content) < 50:
        logger.debug("Skipping %s — content too short (%d chars)", url, len(raw_content))
        return None

    logger.info("Scraped %s — %d chars", url, len(raw_content))
    return {
        "url": url,
        "raw_content": raw_content,
        "image_urls": image_urls,
    }


async def scrape_urls(urls: list[str], cfg: Config) -> list[dict]:
    """Scrape a list of URLs concurrently and return non-empty results.

    Uses a single shared ``httpx.AsyncClient`` (connection pool reuse) and
    a semaphore to cap concurrent requests at ``cfg.MAX_SCRAPER_WORKERS``.

    Args:
        urls: URLs to scrape.
        cfg:  Researcher config providing ``MAX_SCRAPER_WORKERS``,
              ``SCRAPER``, and ``BROWSE_CHUNK_MAX_LENGTH``.

    Returns:
        List of dicts for URLs that yielded content::

            [
                {
                    "url": "https://example.com",
                    "raw_content": "cleaned plain text …",
                    "image_urls": '["https://example.com/img.jpg"]',
                },
                …
            ]

        URLs that time out, return errors, or produce empty content are
        silently dropped (a debug/warning log entry is emitted for each).
    """
    if not urls:
        return []

    sem = asyncio.Semaphore(cfg.MAX_SCRAPER_WORKERS)
    timeout = httpx.Timeout(
        connect=_REQUEST_TIMEOUT,
        read=_REQUEST_TIMEOUT * 2,  # pages can be slow to stream
        write=_REQUEST_TIMEOUT,
        pool=_REQUEST_TIMEOUT,
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as session:
        tasks = [_scrape_one(url, session, sem, cfg) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    return [r for r in results if r is not None]
