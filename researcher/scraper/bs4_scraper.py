import json
import logging
import os
import re

import httpx
from bs4 import BeautifulSoup, Tag  # Tag used in _find_content_node type hint

from .base import BaseScraper

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Tags whose entire subtree is boilerplate — removed before text extraction.
_NOISE_TAGS = [
    "script", "style", "noscript",
    "nav", "footer", "header", "aside",
    "iframe", "svg", "figure",
]

# CSS class fragments that indicate promotional / structural noise.
# Deliberately conservative to avoid false positives:
#   - "ad" was dropped because "header" contains "ad" as a substring, which
#     would decompose the <html> tag and wipe the entire page.
#   - "menu" was dropped because Wikipedia encodes feature flags in <html>
#     class names that contain "menu" (e.g. "vector-feature-main-menu-disabled").
# Structural tags (html, body, main, article) are never removed even if a
# fragment appears in their class list.
_NOISE_CLASS_FRAGMENTS = [
    "advert", "advertisement",
    "cookie-banner", "cookie-notice",
    "popup", "modal",
    "navbar", "sidebar",
    "banner", "promo", "newsletter",
    "share-bar", "sharing",
    "related-posts", "recommended",
]

# These tags anchor the document structure and must never be decomposed by
# class-based removal, even if a fragment accidentally matches their classes.
_PROTECTED_TAGS = frozenset({"html", "body", "main", "article", "section", "div"})

# Ordered list of (tag, attrs) pairs tried when hunting for the main content.
# The first match wins.
_CONTENT_SELECTORS: list[tuple[str, dict]] = [
    ("article", {}),
    ("main", {}),
    ("div", {"class": "content"}),
    ("div", {"class": "main-content"}),
    ("div", {"class": "post-content"}),
    ("div", {"class": "article-body"}),
    ("div", {"class": "entry-content"}),
    ("div", {"id": "content"}),
    ("div", {"id": "main"}),
    ("div", {"role": "main"}),
]

_DEFAULT_CHUNK_MAX = 8192  # fallback if env var / config not available


def _get_chunk_max() -> int:
    try:
        return int(os.getenv("BROWSE_CHUNK_MAX_LENGTH", _DEFAULT_CHUNK_MAX))
    except (ValueError, TypeError):
        return _DEFAULT_CHUNK_MAX



def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove boilerplate tags and noise-class elements in-place.

    Uses individual ``soup.select()`` calls for class-based removal so that
    each query reflects the current tree state.  This avoids the decomposed-
    child problem that arises when iterating a stale ``find_all`` list.
    """
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # CSS attribute selector re-queries the live tree each time, so already-
    # decomposed parents don't leave dangling children in our iteration list.
    # Protected structural tags are skipped even if a fragment matches their
    # class list (e.g. Wikipedia puts feature flags on the <html> element).
    for frag in _NOISE_CLASS_FRAGMENTS:
        for tag in soup.select(f'[class*="{frag}"]'):
            if tag.name not in _PROTECTED_TAGS and tag.parent is not None:
                tag.decompose()

    return soup


def _find_content_node(soup: BeautifulSoup) -> Tag | None:
    """Return the most specific content container, or None for full body."""
    for tag_name, attrs in _CONTENT_SELECTORS:
        node = soup.find(tag_name, attrs)
        if node and isinstance(node, Tag):
            return node
    return soup.find("body")


def _extract_text(node: Tag) -> str:
    """Convert a BeautifulSoup node to clean plain text."""
    raw = node.get_text(separator="\n")
    lines = [line.strip() for line in raw.splitlines()]
    non_empty = [line for line in lines if line]
    # Collapse runs of 3+ blank-ish lines (single-word lines etc.)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(non_empty))
    return cleaned.strip()


def _extract_images(soup: BeautifulSoup, base_url: str) -> str:
    """Return a JSON array of absolute image src URLs."""
    from urllib.parse import urljoin, urlparse

    srcs: list[str] = []
    for img in soup.find_all("img", src=True):
        src: str = img["src"].strip()
        if not src or src.startswith("data:"):
            continue
        absolute = urljoin(base_url, src)
        if urlparse(absolute).scheme in ("http", "https"):
            srcs.append(absolute)
    return json.dumps(srcs)


class BeautifulSoupScraper(BaseScraper):
    """Scrape a page with httpx + BeautifulSoup4.

    Boilerplate removal pipeline:
      1. Fetch page via the shared ``AsyncClient`` with a realistic User-Agent.
      2. Parse with ``html.parser`` (no extra binary dependencies).
      3. Decompose noise tags (script, style, nav, footer …) and noise-class
         elements (ads, popups, menus …).
      4. Find the main content node using a priority list of CSS selectors,
         falling back to ``<body>``.
      5. Extract plain text, collapse whitespace, truncate to
         ``BROWSE_CHUNK_MAX_LENGTH`` characters.
      6. Collect ``<img>`` src URLs and return them as a JSON array string.

    On any error (timeout, HTTP error, decode error) returns ``("", "[]")``.
    """

    async def scrape(self) -> tuple[str, str]:
        try:
            response = await self.session.get(
                self.url,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            )
            response.raise_for_status()

            encoding = response.encoding or "utf-8"
            html = response.content.decode(encoding, errors="replace")

        except httpx.TimeoutException:
            logger.warning("Timeout scraping %s", self.url)
            return "", "[]"
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP %s for %s", exc.response.status_code, self.url)
            return "", "[]"
        except httpx.HTTPError as exc:
            logger.warning("HTTP error scraping %s: %s", self.url, exc)
            return "", "[]"
        except UnicodeDecodeError as exc:
            logger.warning("Decode error for %s: %s", self.url, exc)
            return "", "[]"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error scraping %s: %s", self.url, exc)
            return "", "[]"

        try:
            soup = BeautifulSoup(html, "html.parser")
            image_urls_json = _extract_images(soup, self.url)

            soup = _clean_soup(soup)
            content_node = _find_content_node(soup)

            if content_node is None:
                return "", "[]"

            text = _extract_text(content_node)
            chunk_max = _get_chunk_max()
            text = text[:chunk_max]

            return text, image_urls_json

        except Exception as exc:  # noqa: BLE001
            logger.warning("Parse error for %s: %s", self.url, exc)
            return "", "[]"
