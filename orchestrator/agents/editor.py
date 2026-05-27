import asyncio
import json
import logging
from datetime import date

from json_repair import repair_json

from orchestrator.agents.utils.llms import call_model
from orchestrator.agents.utils.views import (
    format_sections_table,
    log_research_progress,
    print_agent_output,
)
from orchestrator.state import DraftState, ResearchState

logger = logging.getLogger(__name__)


class EditorAgent:
    """Orchestrates report planning and parallel section research.

    Two distinct responsibilities depending on which graph node calls it:

    1. ``plan_research()`` — generates the report outline (section titles)
       from the initial broad research context.
    2. ``run_parallel_research()`` — spawns one sub-graph per section and
       runs them all concurrently, collecting approved drafts.
    """

    def __init__(self, websocket=None) -> None:
        self.websocket = websocket

    # ------------------------------------------------------------------
    # Node 1: plan_research — called by 'planner' node in main graph
    # ------------------------------------------------------------------

    async def plan_research(self, state: ResearchState) -> dict:
        """Generate the report outline from initial research context.

        Uses the STRATEGIC_LLM to produce exactly ``max_sections`` section
        titles that together form a comprehensive, non-overlapping answer to
        the research query.  Introduction, Conclusion, and References are
        intentionally excluded — the writer agent handles those separately.

        Args:
            state: ResearchState; reads ``task`` and ``initial_research``.

        Returns:
            ``{"sections": list[str], "title": str, "date": str}``
        """
        from researcher.config import Config

        task = state["task"]
        query = task["query"]
        initial_research = state["initial_research"]
        max_sections = task.get("max_sections", 3)

        print_agent_output(f"Planning research outline for: {query}", "EDITOR")

        prompt = f"""You are a research editor planning a structured report.

Research query: {query}

Initial research context:
{initial_research[:3000]}

Plan a report outline with exactly {max_sections} main sections.

Rules:
- Each section title must be specific and non-overlapping
- Do NOT include Introduction, Conclusion, References, or Summary as sections
- Sections should together form a comprehensive answer to the query
- Make section titles concise (5-10 words each)

Respond with a JSON object:
{{
  "title": "Full report title here",
  "sections": ["Section 1 Title", "Section 2 Title", "Section 3 Title"]
}}"""

        cfg = Config()
        response = await call_model(
            prompt=prompt,
            model=task.get("model", cfg.STRATEGIC_LLM),
            response_format="json",
            max_tokens=cfg.STRATEGIC_TOKEN_LIMIT,
        )

        try:
            parsed = json.loads(repair_json(response))
            sections = parsed.get("sections", [])
            title = parsed.get("title", query)
        except Exception as exc:
            logger.warning("Failed to parse section outline: %s — using fallback", exc)
            sections = [
                f"Overview of {query}",
                f"Key Developments in {query}",
                f"Future Implications of {query}",
            ]
            title = query

        print(format_sections_table(sections))

        return {
            "sections": sections,
            "title": title,
            "date": date.today().strftime("%B %d, %Y"),
        }

    # ------------------------------------------------------------------
    # Node 2: run_parallel_research — called by 'researcher' node
    # ------------------------------------------------------------------

    async def run_parallel_research(self, state: ResearchState) -> dict:
        """Spawn one sub-graph per section and run them all concurrently.

        Each section gets its own isolated :class:`DraftState` instance,
        which prevents race conditions between parallel executions.
        After all sub-graphs complete, approved drafts and source URLs are
        collected into the shared ResearchState.

        Args:
            state: ResearchState; reads ``task`` and ``sections``.

        Returns:
            ``{"research_data": list[dict], "sources": list[str]}``
        """
        from orchestrator.graph import build_section_subgraph

        task = state["task"]
        sections = state["sections"]

        print_agent_output(
            f"Spawning {len(sections)} parallel sub-graphs...",
            "EDITOR",
        )

        initial_draft_states = [
            DraftState(
                task=task,
                topic=section,
                draft="",
                review=None,
                revision_notes="",
                guidelines=task.get("guidelines", []),
            )
            for section in sections
        ]

        subgraph = build_section_subgraph()

        completed_drafts = await asyncio.gather(
            *(subgraph.ainvoke(draft_state) for draft_state in initial_draft_states),
            return_exceptions=True,
        )

        research_data: list[dict] = []
        all_sources: list[str] = []

        for section, result in zip(sections, completed_drafts):
            if isinstance(result, Exception):
                log_research_progress(section, "failed", str(result))
                logger.error("Section %r failed: %s", section, result)
                continue
            research_data.append({
                "section": section,
                "draft": result.get("draft", ""),
                "sources": [],
            })
            log_research_progress(section, "approved")

        print_agent_output(
            f"All {len(research_data)}/{len(sections)} sections complete.",
            "EDITOR",
        )

        return {
            "research_data": research_data,
            "sources": list(set(all_sources)),
        }
