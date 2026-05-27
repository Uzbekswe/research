"""Research context assembly for the report-writing step.

This module is the bridge between the scraping / summarisation layer and the
final LLM report call.  It takes all accumulated source dicts and produces a
single context string that fits within the SMART_LLM token budget.

Layer 1 (current implementation)
---------------------------------
Simple length-based filtering: sources are concatenated in insertion order
until the estimated token budget is exhausted.  Token count is approximated
as ``len(text) / 4`` (the standard GPT-family heuristic).

Layer 3 (planned upgrade)
--------------------------
Replace the length-based filter with embedding-based similarity ranking:

  1. Embed the research *query* and each source summary / chunk.
  2. Compute cosine similarity between query embedding and each chunk.
  3. Rank chunks by similarity score, then fill the token budget greedily
     from the top of the ranked list.

This ensures the LLM receives the *most relevant* context rather than just
the *first* context, which is especially important for long research runs
that accumulate many sources across multiple search iterations.

The vector store (``researcher/vector_store/``) and embeddings layer
(``researcher/embeddings/``) will supply the infrastructure for Layer 3.

Modelled after gpt_researcher/context/compression.py.
"""

import logging

from researcher.config import Config

logger = logging.getLogger(__name__)

# Chars-per-token approximation (GPT-family heuristic: ~4 chars per token).
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate the token count for *text* using the 4-chars-per-token rule."""
    return len(text) // _CHARS_PER_TOKEN


def _format_source(source: dict) -> str:
    """Render a single source dict as an attributed context block.

    Uses the ``summary`` field when available (shorter, query-focused),
    falling back to ``content`` when the source was not summarised.

    Args:
        source: Dict with keys ``"url"``, ``"content"``, ``"summary"``.

    Returns:
        Formatted string ending with a ``---`` divider, or ``""`` when both
        ``summary`` and ``content`` are empty (so the caller can skip it).
    """
    url = source.get("url", "unknown")
    text = (source.get("summary") or source.get("content", "")).strip()
    if not text:
        return ""
    return f"Source: {url}\n{text}\n\n---\n\n"


async def get_research_context(
    sources: list[dict],
    query: str,
    cfg: Config,
) -> str:
    """Assemble a token-budgeted context string from all research sources.

    Layer 1 implementation: concatenate sources in insertion order, stopping
    when the estimated token count would exceed the budget.

    # Layer 3: replace this with embedding-based similarity filtering so that
    # the most query-relevant chunks fill the budget instead of the first N.

    Args:
        sources: List of source dicts, each with keys:
                 ``"url"`` (str), ``"content"`` (str), ``"summary"`` (str).
                 Typically the output of :meth:`ResearchMemory.get_context`.
        query:   The research question (unused in Layer 1; will drive
                 similarity ranking in Layer 3).
        cfg:     Researcher configuration.  ``cfg.SMART_TOKEN_LIMIT`` sets
                 the reference budget; the hard cap is ``3 ×`` that value to
                 accommodate the full context window.

    Returns:
        A single string of concatenated, attributed source blocks ready to be
        inserted into the report-generation prompt.  Empty string if
        *sources* is empty.
    """
    if not sources:
        return ""

    # Hard token cap: 3× SMART_TOKEN_LIMIT covers the full context window
    # while leaving room for the system prompt and the generated report.
    token_budget = cfg.SMART_TOKEN_LIMIT * 3
    char_budget = token_budget * _CHARS_PER_TOKEN

    logger.debug(
        "Building context: %d sources, budget ≈ %d tokens (%d chars)",
        len(sources),
        token_budget,
        char_budget,
    )

    blocks: list[str] = []
    total_chars = 0

    for source in sources:
        block = _format_source(source)
        if not block:
            continue  # skip sources with no usable text

        block_chars = len(block)

        if total_chars + block_chars > char_budget:
            logger.debug(
                "Token budget reached after %d/%d sources (~%d tokens used)",
                len(blocks),
                len(sources),
                _estimate_tokens(str(total_chars)),
            )
            break

        blocks.append(block)
        total_chars += block_chars

    context = "".join(blocks)

    # Belt-and-suspenders hard truncation in case individual blocks are huge.
    if len(context) > char_budget:
        context = context[:char_budget]
        logger.debug("Context hard-truncated to %d chars", char_budget)

    logger.info(
        "Context assembled: %d/%d sources, ~%d tokens",
        len(blocks),
        len(sources),
        _estimate_tokens(context),
    )
    return context
