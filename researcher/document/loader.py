"""Document loader stub — Layer 2 placeholder.

In Layer 2 this module will scan ``path`` for supported file types
(.pdf, .docx, .txt, .md) using python-docx / PyMuPDF and return their
content as source dicts compatible with ResearchMemory.add_source().
"""

import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads local documents from a directory path.

    TODO (Layer 2): implement actual file-type dispatching:
      - .pdf  → PyMuPDF (fitz) page-by-page text extraction
      - .docx → python-docx paragraph extraction
      - .txt / .md → plain read()
      Each document becomes one source dict: {"url": file_path, "content": text, "summary": ""}
    """

    async def load_documents(self, path: str) -> list[dict]:
        """Load all supported documents from *path* and return source dicts.

        Args:
            path: Directory (or single file) path to load from.

        Returns:
            List of ``{"url": str, "content": str, "summary": str}`` dicts.
            Currently always returns ``[]`` until Layer 2 is implemented.
        """
        # TODO (Layer 2): walk `path`, dispatch by extension, parse content
        logger.warning(
            "DocumentLoader.load_documents() is not yet implemented (Layer 2). "
            "Returning empty list for path: %s",
            path,
        )
        return []
