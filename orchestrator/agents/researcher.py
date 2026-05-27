from orchestrator.agents.utils.views import log_research_progress, print_agent_output
from orchestrator.state import DraftState, ResearchState


class ResearchAgent:
    """Bridge between the orchestrator graph and the researcher RAG package.

    The orchestrator has no knowledge of how RAG works internally.
    It only calls this agent, which wraps DeepResearcher and returns
    plain dicts that flow back into the shared graph state.
    """

    def __init__(self) -> None:
        pass

    async def run_initial_research(self, state: ResearchState) -> dict:
        """Broad initial research sweep on the full query.

        Called by the ``browser`` node in the main graph.  Gives the editor
        enough context to plan coherent section titles before deep per-section
        research begins.

        Args:
            state: Current ResearchState; reads ``state["task"]``.

        Returns:
            ``{"initial_research": str}`` — assembled context string.
        """
        task = state["task"]
        query = task["query"]

        print_agent_output(f"Running initial research on: {query}", "RESEARCHER")

        from researcher import DeepResearcher

        researcher = DeepResearcher(
            query=query,
            report_type="research_report",
            report_source=task.get("source", "web"),
        )

        await researcher.conduct_research()
        initial_research = researcher.get_research_context()

        print_agent_output(
            f"Initial research complete.\n"
            f"Sources: {len(researcher.get_source_urls())}\n"
            f"Context: ~{len(initial_research) // 4} tokens",
            "RESEARCHER",
        )

        return {"initial_research": initial_research}

    async def run_depth_research(self, draft_state: DraftState) -> dict:
        """Deep research on one specific section topic.

        Called by the ``researcher`` node in the per-section sub-graph.
        Each call receives its own isolated DraftState, so concurrent
        execution across sections has no shared mutable state.

        Args:
            draft_state: Isolated state for one section; reads ``task``
                         and ``topic``.

        Returns:
            ``{"draft": str, "revision_notes": str, "review": None}``
            where ``review=None`` signals the section is ready for review.
        """
        task = draft_state["task"]
        topic = draft_state["topic"]
        parent_query = task["query"]

        log_research_progress(topic, "started")

        from researcher import DeepResearcher

        researcher = DeepResearcher(
            query=topic,
            report_type="research_report",
            parent_query=parent_query,
            report_source=task.get("source", "web"),
        )

        await researcher.conduct_research()
        draft = await researcher.write_report()
        sources = researcher.get_source_urls()

        log_research_progress(topic, "complete", f"{len(sources)} sources")

        return {
            "draft": draft,
            "revision_notes": "",
            "review": None,
        }
