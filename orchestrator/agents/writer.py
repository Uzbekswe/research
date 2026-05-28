import logging

from orchestrator.agents.utils.llms import call_model
from orchestrator.agents.utils.views import print_agent_output
from orchestrator.state import ResearchState

logger = logging.getLogger(__name__)


class WriterAgent:
    """Compiles all approved section drafts into a final cohesive report.

    Writes the introduction and conclusion via LLM, then collects and
    deduplicates all source URLs.  No assembly into a final string happens
    here — that is PublisherAgent's job.
    """

    async def run(self, state: ResearchState) -> dict:
        from researcher.config import Config

        cfg = Config()
        task = state["task"]
        research_data = state["research_data"]
        title = state.get("title", task["query"])

        print_agent_output("Compiling final report...", "WRITER")

        sections_text = "\n\n".join(
            f"## {item['section']}\n\n{item['draft']}"
            for item in research_data
            if item.get("draft")
        )

        # OPUS FIX (3B): use the centralised prompt templates instead of inline strings.
        from researcher.prompts import (
            get_report_conclusion_prompt,
            get_report_introduction_prompt,
        )

        # ── Introduction ──────────────────────────────────────────────
        intro_prompt = get_report_introduction_prompt(
            title=title,
            query=task["query"],
            section_titles=[item["section"] for item in research_data],
        )

        introduction = await call_model(
            prompt=intro_prompt,
            model=task.get("model", cfg.SMART_LLM),
            max_tokens=600,
            temperature=0.5,
        )

        # ── Conclusion ────────────────────────────────────────────────
        conclusion_prompt = get_report_conclusion_prompt(
            title=title,
            query=task["query"],
            sections_text=sections_text,
        )

        conclusion = await call_model(
            prompt=conclusion_prompt,
            model=task.get("model", cfg.SMART_LLM),
            max_tokens=600,
            temperature=0.5,
        )

        # ── Sources ───────────────────────────────────────────────────
        all_sources: list[str] = []
        for item in research_data:
            all_sources.extend(item.get("sources", []))
        # OPUS FIX: also fold in sources already accumulated on the shared state
        # by the editor's run_parallel_research, so we don't lose them.
        all_sources.extend(state.get("sources", []) or [])
        # dict.fromkeys preserves insertion order while deduplicating
        all_sources = list(dict.fromkeys(all_sources))

        # OPUS FIX (3F): populate table_of_contents and headers so ResearchState
        # is fully filled in — matches GPT Researcher's writer contract.
        toc_lines = ["## Table of Contents", ""]
        for i, item in enumerate(research_data, 1):
            anchor = item["section"].lower().replace(" ", "-")
            toc_lines.append(f"{i}. [{item['section']}](#{anchor})")
        table_of_contents = "\n".join(toc_lines)

        headers = {
            "introduction": "## Introduction",
            "table_of_contents": "## Table of Contents",
            "conclusion": "## Conclusion",
            "references": "## References",
        }

        print_agent_output("Report compilation complete.", "WRITER")

        return {
            "introduction": introduction,
            "conclusion": conclusion,
            "sources": all_sources,
            "table_of_contents": table_of_contents,
            "headers": headers,
        }
