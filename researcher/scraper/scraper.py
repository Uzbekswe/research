"""Main entry point for the scraping layer.

Call ``scrape_urls()`` with a list of URLs and a ``Config`` instance.
It manages the shared HTTP session, concurrency cap, retry logic, and
result aggregation.
"""

import asyncio
import logging
import time

import httpx

from researcher.config import Config

from . import get_scraper

logger = logging.getLogger(__name__)

_PER_URL_TIMEOUT = 8.0   # seconds per individual URL request
_TOTAL_TIMEOUT = 60      # seconds max for the entire scrape_urls() call
_MIN_CONTENT_LENGTH = 100  # chars — shorter results are redirect/error pages


async def scrape_urls(urls: list[str], cfg: Config) -> list[dict]:
    """Scrape a list of URLs concurrently and return non-empty results.

    Uses a single shared ``httpx.AsyncClient`` (connection pool reuse),
    a semaphore to cap concurrent requests at ``cfg.MAX_SCRAPER_WORKERS``,
    per-URL retry with exponential back-off, and a 60-second hard timeout
    over the entire batch.

    Args:
        urls: URLs to scrape.
        cfg:  Researcher config providing ``MAX_SCRAPER_WORKERS`` and ``SCRAPER``.

    Returns:
        List of dicts for URLs that yielded ≥100 chars of content::

            [{"url": str, "raw_content": str, "image_urls": str}, ...]

        URLs that time out, error on all 3 attempts, or produce short content
        are silently dropped (warning logs are emitted for each).
    """
    if not urls:
        return []

    semaphore = asyncio.Semaphore(cfg.MAX_SCRAPER_WORKERS)
    timeout = httpx.Timeout(
        connect=_PER_URL_TIMEOUT,
        read=_PER_URL_TIMEOUT,
        write=_PER_URL_TIMEOUT,
        pool=_PER_URL_TIMEOUT,
    )
    ScraperClass = get_scraper(cfg.SCRAPER)

    # Results written here as coroutines complete — safe because asyncio is
    # single-threaded and list.append() has no yield point.
    collected: list[dict] = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as session:

        async def scrape_with_retry(url: str) -> None:
            async with semaphore:
                for attempt in range(3):
                    t0 = time.monotonic()
                    try:
                        scraper = ScraperClass(url=url, session=session)
                        raw_content, image_urls = await scraper.scrape()
                        elapsed = time.monotonic() - t0

                        if raw_content:
                            logger.debug(
                                "Scraped %s in %.2fs (%d chars)", url, elapsed, len(raw_content)
                            )
                            collected.append(
                                {"url": url, "raw_content": raw_content, "image_urls": image_urls}
                            )
                        return  # success or intentionally empty — don't retry

                    except Exception as exc:
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (attempt + 1))  # 0.5s, 1.0s back-off
                        else:
                            logger.warning(
                                "⚠️ Failed to scrape %s after 3 attempts: %s", url, exc
                            )

        tasks = [asyncio.create_task(scrape_with_retry(url)) for url in urls]
        try:
            async with asyncio.timeout(_TOTAL_TIMEOUT):
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning(
                "⏱ Scraping timed out after %ds — returning %d partial results",
                _TOTAL_TIMEOUT, len(collected),
            )
            for task in tasks:
                if not task.done():
                    task.cancel()

    # Drop results that are too short to be useful (redirect pages, error pages).
    return [r for r in collected if len(r["raw_content"]) >= _MIN_CONTENT_LENGTH]
