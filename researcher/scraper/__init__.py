from .base import BaseScraper
from .bs4_scraper import BeautifulSoupScraper

# Add PlaywrightScraper here later for JS-rendered pages

_REGISTRY: dict[str, type[BaseScraper]] = {
    "bs4": BeautifulSoupScraper,
}


def get_scraper(scraper_name: str) -> type[BaseScraper]:
    """Return the scraper *class* for the given name string.

    Args:
        scraper_name: Currently ``"bs4"`` for BeautifulSoupScraper.

    Returns:
        The scraper class (not an instance).  Instantiate with
        ``get_scraper("bs4")(url="...", session=async_client)``.

    Raises:
        ValueError: If ``scraper_name`` is not a registered scraper.
    """
    cls = _REGISTRY.get(scraper_name.lower().strip())
    if cls is None:
        supported = ", ".join(f'"{k}"' for k in sorted(_REGISTRY))
        raise ValueError(
            f"Unknown scraper {scraper_name!r}. Supported scrapers: {supported}"
        )
    return cls


__all__ = [
    "BaseScraper",
    "BeautifulSoupScraper",
    "get_scraper",
]
