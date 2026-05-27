"""DeepResearcher — public API of the researcher package.

All consumer code interacts with this class exclusively; the sub-packages
(actions, scraper, retrievers, prompts, …) are implementation details.

Modelled after gpt_researcher/agent.py (GPTResearcher).
"""

import asyncio
import json
import logging

from researcher.actions.query_processing import get_sub_queries
from researcher.actions.report_generation import (
    summarize_url,
    write_report as _write_report,
    write_report_conclusion as _write_report_conclusion,
)
from researcher.actions.web_scraping import browse_web_sources, search_and_scrape
from researcher.config import Config
from researcher.context.context_manager import get_research_context
from researcher.memory.research_memory import ResearchMemory
from researcher.prompts import get_agent_role_prompt

logger = logging.getLogger(__name__)


class DeepResearcher:
    """Autonomous research agent that searches, scrapes, summarises, and writes reports.

    Typical usage::

        researcher = DeepResearcher(query="What is quantum computing?")
        await researcher.conduct_research()
        report = await researcher.write_report()

    Args:
        query:         The research question or task.
        report_type:   One of ``"research_report"``, ``"outline_report"``,
                       ``"resource_report"``.
        report_source: ``"web"`` (default) or ``"local"``.
        source_urls:   If provided, skip search and scrape these URLs directly.
        document_urls: Additional document URLs (used in local-source mode).
        config_path:   Path to a JSON config file that overrides env vars.
        websocket:     Optional WebSocket for streaming progress updates.
        agent:         Pre-determined agent type string (skips auto-detection).
        role:          Pre-determined agent role / system prompt.
        parent_query:  Parent question when this is a sub-topic run.
        subtopic:      Current sub-topic label (used in multi-level reports).
        headers:       Extra HTTP headers or config overrides as a dict.
        max_subtopics: Maximum number of subtopics to generate.
    """

    def __init__(
        self,
        query: str,
        report_type: str = "research_report",
        report_source: str = "web",
        source_urls: list[str] | None = None,
        document_urls: list[str] | None = None,
        config_path: str | None = None,
        websocket=None,
        agent: str | None = None,
        role: str | None = None,
        parent_query: str = "",
        subtopic: str = "",
        headers: dict | None = None,
        max_subtopics: int = 3,
    ) -> None:
        self.query = query
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls or []
        self.document_urls = document_urls or []
        self.websocket = websocket
        self.agent = agent
        self.role = role or get_agent_role_prompt(query)
        self.parent_query = parent_query
        self.subtopic = subtopic
        self.headers = headers or {}
        self.max_subtopics = max_subtopics

        # Config — reads env vars, then overrides from JSON file if given.
        self.cfg = Config(config_path)

        # In-process state for this research run.
        self.memory = ResearchMemory()

        # context is a string after conduct_research(); empty until then.
        self.context: str = ""

        # Mirrors GPT Researcher for API compatibility.
        self.research_costs: float = 0.0
        self.verbose: bool = True

    # ------------------------------------------------------------------
    # Core research workflow
    # ------------------------------------------------------------------

    async def conduct_research(self) -> str:
        """Run the full research loop and return the assembled context string.

        Steps
        -----
        1. Determine agent role (already set in ``__init__``; can be overridden).
        2. If ``report_source == "local"``, stub — document loading is Layer 2.
        3. Resolve sources: either caller-supplied URLs or web search per sub-query.
        4. Scrape, summarise, and deduplicate all sources concurrently.
        5. Assemble context via :func:`~researcher.context.context_manager.get_research_context`.
        6. Store result in ``self.context`` and return it.

        Returns:
            The assembled context string ready to be passed to :meth:`write_report`.
        """
        logger.info("Starting research for: %r", self.query)

        # ── Step 2: local source mode ─────────────────────────────────────
        if self.report_source == "local":
            logger.warning(
                "Local document mode is not yet implemented (Layer 2). "
                "Falling back to web search."
            )

        # ── Step 3: decide where to get URLs from ────────────────────────
        if self.source_urls:
            # User supplied explicit URLs — skip the search step entirely.
            logger.info("Using %d caller-supplied source URLs", len(self.source_urls))
            all_scraped = await browse_web_sources(
                query=self.query,
                urls=self.source_urls,
                cfg=self.cfg,
                websocket=self.websocket,
            )
            await self._ingest_scraped(all_scraped)
        else:
            # ── Step 3a: generate sub-queries ─────────────────────────────
            sub_queries = await get_sub_queries(
                query=self.query,
                agent_role_prompt=self.role,
                cfg=self.cfg,
                parent_query=self.parent_query,
                report_type=self.report_type,
                cost_callback=self.add_costs,
            )
            # Always include the original query so we don't miss direct hits.
            if self.query not in sub_queries:
                sub_queries = [self.query] + sub_queries
            logger.info("Generated %d sub-queries: %s", len(sub_queries), sub_queries)

            # ── Step 4: search + scrape all sub-queries concurrently ──────
            scrape_tasks = [
                search_and_scrape(q, self.cfg, self.websocket) for q in sub_queries
            ]
            results_per_query: list[list[dict]] = await asyncio.gather(*scrape_tasks)

            for sub_query, scraped_list in zip(sub_queries, results_per_query):
                logger.debug(
                    "Sub-query %r returned %d scraped sources", sub_query, len(scraped_list)
                )
                await self._ingest_scraped(scraped_list)

        # ── Step 5: build context string ─────────────────────────────────
        sources = self.memory.get_context(max_sources=20)
        self.context = await get_research_context(
            sources=sources,
            query=self.query,
            cfg=self.cfg,
        )

        logger.info(
            "Research complete — %d sources, context ~%d tokens",
            len(sources),
            len(self.context) // 4,
        )
        return self.context

    async def _ingest_scraped(self, scraped_list: list[dict]) -> None:
        """Summarise new (unvisited) scraped sources and add them to memory.

        Summarisation calls are fired concurrently for all new sources in the
        batch, then results are written to memory sequentially (asyncio is
        single-threaded so there are no race conditions, but sequential writes
        keep the visited-URL dedup logic clean).

        Args:
            scraped_list: Output of :func:`~researcher.actions.web_scraping.search_and_scrape`.
        """
        new_sources = [
            s for s in scraped_list if not self.memory.is_visited(s["url"])
        ]
        if not new_sources:
            return

        # Summarise all new sources concurrently.
        summarise_tasks = [
            summarize_url(
                url=s["url"],
                content=s["raw_content"],
                query=self.query,
                cfg=self.cfg,
                cost_callback=self.add_costs,
            )
            for s in new_sources
        ]
        summaries: list[str] = await asyncio.gather(*summarise_tasks)

        # Write to memory sequentially (no yield points → no races).
        for source, summary in zip(new_sources, summaries):
            self.memory.add_source(
                url=source["url"],
                content=source["raw_content"],
                summary=summary,
            )
            # Collect image URLs found during scraping.
            try:
                imgs: list[str] = json.loads(source.get("image_urls", "[]"))
                self.memory.add_images(imgs)
            except (json.JSONDecodeError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Report writing
    # ------------------------------------------------------------------

    async def write_report(self, custom_prompt: str = "") -> str:
        """Write the final research report from the accumulated context.

        Must be called after :meth:`conduct_research` (or after manually
        setting ``self.context``).

        Args:
            custom_prompt: Optional prompt that replaces the standard report
                           template.  The context is still appended.

        Returns:
            Completed Markdown report string.
        """
        return await _write_report(
            query=self.query,
            context=self.memory.get_context(),
            cfg=self.cfg,
            report_type=self.report_type,
            agent_role_prompt=self.role,
            websocket=self.websocket,
            cost_callback=self.add_costs,
            custom_prompt=custom_prompt,
        )

    async def write_report_conclusion(self, report: str) -> str:
        """Generate a 2–3 paragraph conclusion for an existing report body.

        Args:
            report: The report text produced by :meth:`write_report`.

        Returns:
            Conclusion section as a Markdown string.
        """
        return await _write_report_conclusion(
            query=self.query,
            report_body=report,
            cfg=self.cfg,
            cost_callback=self.add_costs,
        )

    # ------------------------------------------------------------------
    # Subtopics
    # ------------------------------------------------------------------

    async def get_subtopics(self) -> list[str]:
        """Return a list of subtopic strings for detailed report generation.

        Uses the STRATEGIC_LLM with ``report_type="subtopic_report"`` so that
        the prompt frames queries as section headers rather than search queries.

        Returns:
            List of subtopic strings (up to ``cfg.MAX_SUBTOPICS``).
        """
        subtopics = await get_sub_queries(
            query=self.query,
            agent_role_prompt=self.role,
            cfg=self.cfg,
            parent_query=self.parent_query,
            report_type="subtopic_report",
            cost_callback=self.add_costs,
        )
        return subtopics[: self.cfg.MAX_SUBTOPICS]

    # ------------------------------------------------------------------
    # Getters — mirror GPT Researcher API exactly
    # ------------------------------------------------------------------

    def get_research_context(self) -> str:
        """Return the assembled context string (set by :meth:`conduct_research`)."""
        return self.context

    def get_source_urls(self) -> list[str]:
        """Return all visited source URLs in scrape order."""
        return self.memory.get_source_urls()

    def get_research_sources(self) -> list[dict]:
        """Return all accumulated source dicts ``{"url", "content", "summary"}``."""
        return self.memory.get_context()

    def get_costs(self) -> float:
        """Return the total accumulated LLM cost for this run."""
        return self.memory.get_costs()

    def get_research_images(self) -> list[str]:
        """Return all image URLs collected from scraped pages."""
        return self.memory.images

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose logging for this instance."""
        self.verbose = verbose

    def add_costs(self, cost) -> None:
        """Accumulate LLM cost.

        Accepts either a ``float`` (direct cost value) or a ``dict`` produced
        by the action-layer cost callbacks (``{"llm": str, "tokens": int, ...}``).
        Dict costs are summed from a ``"cost"`` key if present; otherwise 0.0
        is added (token counts aren't yet converted to currency).

        Args:
            cost: A float cost value, or a dict from a cost_callback.
        """
        if isinstance(cost, dict):
            amount = float(cost.get("cost", 0.0))
        else:
            try:
                amount = float(cost)
            except (TypeError, ValueError):
                logger.warning("add_costs received unexpected type %s; ignoring", type(cost))
                return
        self.memory.add_costs(amount)
        self.research_costs = self.memory.get_costs()
