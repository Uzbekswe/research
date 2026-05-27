from langgraph.graph import END, StateGraph

from orchestrator.agents.editor import EditorAgent
from orchestrator.agents.publisher import PublisherAgent
from orchestrator.agents.researcher import ResearchAgent
from orchestrator.agents.reviewer import ReviewerAgent
from orchestrator.agents.reviser import ReviserAgent
from orchestrator.agents.writer import WriterAgent
from orchestrator.state import DraftState, ResearchState


def build_section_subgraph():
    """Build the per-section sub-graph: researcher → reviewer ↔ reviser loop → END.

    Runs once per report section, concurrently with other sections.
    Each instance has its own isolated DraftState — no shared mutable state.

    Conditional edge:
      ``state["review"] is None``  → END (approved)
      ``state["review"]`` is a str → reviser (needs revision)
    """
    research_agent = ResearchAgent()
    reviewer_agent = ReviewerAgent()
    reviser_agent = ReviserAgent()

    subgraph = StateGraph(DraftState)

    subgraph.add_node("researcher", research_agent.run_depth_research)
    subgraph.add_node("reviewer", reviewer_agent.run)
    subgraph.add_node("reviser", reviser_agent.run)

    subgraph.set_entry_point("researcher")
    subgraph.add_edge("researcher", "reviewer")
    subgraph.add_edge("reviser", "reviewer")

    subgraph.add_conditional_edges(
        "reviewer",
        lambda state: END if state.get("review") is None else "reviser",
        {END: END, "reviser": "reviser"},
    )

    return subgraph.compile()


def build_main_graph(output_dir: str = "./outputs"):
    """Build the main research graph.

    Pipeline:
      browser → planner → researcher (parallel sub-graphs) → writer → publisher

    Each node receives the full ResearchState, writes only the fields it
    owns back into the state, and passes control to the next node.
    No agent calls another agent directly.
    """
    research_agent = ResearchAgent()
    editor_agent = EditorAgent()
    writer_agent = WriterAgent()
    publisher_agent = PublisherAgent(output_dir=output_dir)

    workflow = StateGraph(ResearchState)

    workflow.add_node("browser", research_agent.run_initial_research)
    workflow.add_node("planner", editor_agent.plan_research)
    workflow.add_node("researcher", editor_agent.run_parallel_research)
    workflow.add_node("writer", writer_agent.run)
    workflow.add_node("publisher", publisher_agent.run)

    workflow.set_entry_point("browser")
    workflow.add_edge("browser", "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "publisher")
    workflow.add_edge("publisher", END)

    return workflow.compile()
