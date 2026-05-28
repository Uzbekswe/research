"""DeepResearcher — public API of the researcher package.

All consumer code interacts with this class exclusively; the sub-packages
(actions, scraper, retrievers, prompts, …) are implementation details.

Modelled after gpt_researcher/agent.py (GPTResearcher).
"""

import asyncio
import json
import logging
import time

from researcher.actions.query_processing import get_sub_queries
from researcher.actions.report_generation import (
    summarize_url,
    write_report as _write_report,
    write_report_conclusion as _write_report_conclusion,
)
from researcher.actions.web_scraping import browse_web_sources, search_and_scrape
from researcher.config import Config
from researcher.context.context_manager import build_context_store, get_research_context
from researcher.memory.research_memory import ResearchMemory
from researcher.prompts import get_agent_role_prompt
from researcher.vector_store import MemoryVectorStore

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

        # Set by conduct_research(); accessible via getters below.
        self.sub_queries: list[str] = []
        self.vector_store: MemoryVectorStore | None = None

        # Mirrors GPT Researcher for API compatibility.
        self.research_costs: float = 0.0
        self.verbose: bool = True

    # ------------------------------------------------------------------
    # Core research workflow
    # ------------------------------------------------------------------

    async def conduct_research(self) -> str:
        """Run the full research loop and return the assembled context string."""
        start_time = time.monotonic()
        logger.info("🔍 Starting research for: %s", self.query)

        if self.report_source == "local":
            logger.warning(
                "Local document mode is not yet implemented. Falling back to web search."
            )

        sub_queries: list[str] = []

        if self.source_urls:
            all_scraped = await browse_web_sources(
                query=self.query,
                urls=self.source_urls,
                cfg=self.cfg,
                websocket=self.websocket,
            )
            for result in all_scraped:
                url = result["url"]
                if url not in self.memory.visited_urls:
                    self.memory.visited_urls.add(url)
                    summary = await summarize_url(
                        url=url,
                        content=result["raw_content"],
                        query=self.query,
                        cfg=self.cfg,
                        cost_callback=self.add_costs,
                    )
                    self.memory.add_source_no_visit_check(url, result["raw_content"], summary)
                    try:
                        imgs: list[str] = json.loads(result.get("image_urls", "[]"))
                        self.memory.add_images(imgs)
                    except (json.JSONDecodeError, TypeError):
                        pass
        else:
            # ── 1. Generate sub-queries sequentially ─────────────────────
            sub_queries = await get_sub_queries(
                query=self.query,
                agent_role_prompt=self.role,
                cfg=self.cfg,
                parent_query=self.parent_query,
                report_type=self.report_type,
                cost_callback=self.add_costs,
            )
            if self.query not in sub_queries:
                sub_queries = [self.query] + sub_queries
            logger.info("📋 Generated %d sub-queries: %s", len(sub_queries), sub_queries)

            # ── 2. Process all sub-queries concurrently ───────────────────
            async def process_single_query(sub_query: str) -> int:
                try:
                    # Step 1: scrape all results for this sub-query
                    scraped = await search_and_scrape(sub_query, self.cfg, self.websocket)

                    # Step 2: claim unvisited URLs atomically (no await → no race)
                    new_sources = []
                    for result in scraped:
                        if result.get("raw_content") and result["url"] not in self.memory.visited_urls:
                            self.memory.visited_urls.add(result["url"])
                            new_sources.append(result)

                    if not new_sources:
                        return 0

                    # Step 3: summarize ALL new sources concurrently
                    summarize_tasks = [
                        summarize_url(s["url"], s["raw_content"], sub_query, self.cfg, self.add_costs)
                        for s in new_sources
                    ]
                    summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

                    # Step 4: store results
                    count = 0
                    for source, summary in zip(new_sources, summaries):
                        if isinstance(summary, Exception) or not summary:
                            continue
                        self.memory.add_source_no_visit_check(
                            source["url"], source["raw_content"], summary
                        )
                        try:
                            imgs: list[str] = json.loads(source.get("image_urls", "[]"))
                            self.memory.add_images(imgs)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        count += 1
                    return count

                except Exception as exc:
                    logger.error("Sub-query %r failed: %s", sub_query, exc)
                    return 0

            _gather_start = time.monotonic()
            tasks = [process_single_query(q) for q in sub_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            _gather_elapsed = time.monotonic() - _gather_start

            # ── 3. Log per-sub-query results + parallelism benefit ────────
            total = 0
            for sub_query, result in zip(sub_queries, results):
                count = result if isinstance(result, int) else 0
                total += count
                logger.info("  → '%s': %d sources", sub_query, count)

            _avg_time = 15  # conservative sequential estimate per sub-query (seconds)
            logger.info(
                "⚡ Parallel execution: %d sub-queries ran concurrently", len(sub_queries)
            )
            logger.info(
                "   Sequential estimate: ~%ds | Actual: %.1fs",
                len(sub_queries) * _avg_time, _gather_elapsed,
            )
            logger.info(
                "✅ Research complete. %d sources found across %d sub-queries",
                total, len(sub_queries),
            )

        # ── 4. Layer 3: embedding-based context filtering ────────────────
        self.sub_queries = sub_queries
        raw_sources = self.memory.get_context()

        if raw_sources:
            logger.info("🔢 Building embeddings for %d sources...", len(raw_sources))
            self.vector_store = await build_context_store(raw_sources, self.cfg)
            self.context = await get_research_context(
                query=self.query,
                sub_queries=sub_queries,
                store=self.vector_store,
                cfg=self.cfg,
            )
        else:
            self.context = ""
            logger.info("⚠️ No sources to embed — context is empty")

        elapsed = time.monotonic() - start_time
        logger.info("📊 Context length: ~%d tokens", len(self.context) // 4)
        logger.info("⏱ Research completed in %.1fs", elapsed)
        return self.context

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

    def get_sub_queries(self) -> list[str]:
        """Return the sub-queries generated during :meth:`conduct_research`."""
        return self.sub_queries

    def get_vector_store(self) -> MemoryVectorStore | None:
        """Return the populated vector store, or None before research runs."""
        return self.vector_store

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose logging for this instance."""
        self.verbose = verbose

    def add_costs(self, cost) -> None:
        """Accumulate LLM cost.

        Accepts either a ``float`` (direct cost value) or a ``dict`` from the
        action-layer cost callbacks. The dict shape produced by providers is
        ``{"llm": str, "prompt_tokens": int, "completion_tokens": int}`` — this
        method computes USD from those token counts via the pricing table.
        A pre-computed ``"cost"`` key is honoured when present for forward
        compatibility.

        Args:
            cost: A float cost value, or a dict from a cost_callback.
        """
        # OPUS FIX (3I): actually compute USD from token counts. Before this fix
        # cost_callbacks always passed {"tokens": 0} and the dict branch fell back
        # to 0.0, so get_costs() always returned 0.0.
        from researcher.llm_providers.pricing import estimate_cost

        if isinstance(cost, dict):
            if "cost" in cost:
                amount = float(cost.get("cost", 0.0))
            else:
                model = cost.get("llm", "")
                _, _, model_name = model.partition(":") if ":" in model else ("", "", model)
                amount = estimate_cost(
                    model=model_name,
                    prompt_tokens=int(cost.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(cost.get("completion_tokens", 0) or 0),
                )
        else:
            try:
                amount = float(cost)
            except (TypeError, ValueError):
                logger.warning("add_costs received unexpected type %s; ignoring", type(cost))
                return
        self.memory.add_costs(amount)
        self.research_costs = self.memory.get_costs()
