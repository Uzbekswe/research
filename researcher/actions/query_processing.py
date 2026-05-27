"""Query processing actions.

Handles sub-question generation (planning phase) and any query-level
transformations before retrieval begins.
"""

import logging

import json_repair

from researcher.config import Config
from researcher.llm_providers import get_llm_provider
from researcher.prompts import get_search_queries_prompt

logger = logging.getLogger(__name__)


async def get_sub_queries(
    query: str,
    agent_role_prompt: str,
    cfg: Config,
    parent_query: str = "",
    report_type: str = "research_report",
    cost_callback: callable = None,
) -> list[str]:
    """Generate a list of search sub-queries for the given research question.

    Calls STRATEGIC_LLM with the search-query prompt and parses the returned
    JSON array.  Falls back to SMART_LLM if the strategic model fails so that
    a network hiccup or quota exhaustion never halts the pipeline entirely.

    ``json_repair`` is used to tolerate minor LLM formatting mistakes such as
    trailing commas or missing brackets.

    Args:
        query:             The specific question or sub-topic to plan queries for.
        agent_role_prompt: System prompt that sets the researcher's persona.
        cfg:               Researcher configuration.
        parent_query:      Top-level research question (used for sub-topic framing).
        report_type:       Report type string forwarded to the prompt template.
        cost_callback:     Optional callable invoked with token-usage metadata.
                           Signature: ``cost_callback({"llm": str, "tokens": int})``.
                           Token counts are not yet surfaced by BaseLLMProvider;
                           the callback receives ``tokens=0`` as a placeholder.

    Returns:
        Ordered list of search query strings.  Falls back to ``[query]`` if all
        LLM attempts fail.
    """
    prompt = get_search_queries_prompt(
        query=query,
        parent_query=parent_query,
        report_type=report_type,
        max_iterations=cfg.MAX_ITERATIONS,
    )
    messages = [
        {"role": "system", "content": agent_role_prompt},
        {"role": "user", "content": prompt},
    ]

    response: str = ""

    # Primary: STRATEGIC_LLM
    try:
        provider = get_llm_provider(cfg.STRATEGIC_LLM, cfg.TEMPERATURE, cfg.STRATEGIC_TOKEN_LIMIT)
        response = await provider.get_chat_response(messages, max_tokens=cfg.STRATEGIC_TOKEN_LIMIT)
        if cost_callback:
            cost_callback({"llm": cfg.STRATEGIC_LLM, "tokens": 0})
    except Exception as exc:
        logger.warning("STRATEGIC_LLM failed for sub-query generation: %s — falling back to SMART_LLM", exc)

        # Fallback: SMART_LLM
        try:
            provider = get_llm_provider(cfg.SMART_LLM, cfg.TEMPERATURE, cfg.SMART_TOKEN_LIMIT)
            response = await provider.get_chat_response(messages, max_tokens=cfg.SMART_TOKEN_LIMIT)
            if cost_callback:
                cost_callback({"llm": cfg.SMART_LLM, "tokens": 0})
        except Exception as exc2:
            logger.error("SMART_LLM fallback also failed: %s — returning original query", exc2)
            return [query]

    if not response:
        logger.warning("Empty LLM response for sub-query generation; returning original query")
        return [query]

    try:
        queries = json_repair.loads(response)
        if not isinstance(queries, list):
            raise ValueError(f"Expected a JSON list, got {type(queries).__name__}")
        # Filter out any non-string items and strip whitespace
        return [str(q).strip() for q in queries if q and str(q).strip()]
    except Exception as exc:
        logger.warning("Failed to parse sub-query response %r: %s — returning original query", response[:200], exc)
        return [query]


async def get_similar_written_contents_by_draft_section_titles(
    query: str,
    draft_section_titles: list[str],
    written_contents: list[dict],
    cfg: Config,
) -> list[dict]:
    """Find already-written content sections relevant to the given draft titles.

    TODO: Implement Layer 3 — embed draft_section_titles and written_contents,
    compute cosine similarity via the vector store, and return the top-k most
    similar written sections so the report writer can avoid duplication.

    Args:
        query:                The top-level research question.
        draft_section_titles: Candidate section headers for the next sub-report.
        written_contents:     Previously written sections as list of dicts
                              with keys ``"header"`` and ``"content"``.
        cfg:                  Researcher configuration.

    Returns:
        Subset of ``written_contents`` most relevant to ``draft_section_titles``.
    """
    # TODO: embed + cosine-similarity retrieval (Layer 3)
    return []
