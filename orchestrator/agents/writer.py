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

        # ── Introduction ──────────────────────────────────────────────
        intro_prompt = f"""Write a 2-3 paragraph introduction for a research report.

Report title: {title}
Query: {task['query']}

The report covers these sections:
{chr(10).join(f"- {item['section']}" for item in research_data)}

Write only the introduction paragraphs. No heading needed."""

        introduction = await call_model(
            prompt=intro_prompt,
            model=task.get("model", cfg.SMART_LLM),
            max_tokens=600,
            temperature=0.5,
        )

        # ── Conclusion ────────────────────────────────────────────────
        conclusion_prompt = f"""Write a 2-3 paragraph conclusion for this research report.

Report title: {title}
Query: {task['query']}

Report content summary:
{sections_text[:4000]}

Write only the conclusion paragraphs. No heading needed.
Synthesize key findings. Do not introduce new information."""

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
        # dict.fromkeys preserves insertion order while deduplicating
        all_sources = list(dict.fromkeys(all_sources))

        print_agent_output("Report compilation complete.", "WRITER")

        return {
            "introduction": introduction,
            "conclusion": conclusion,
            "sources": all_sources,
        }
