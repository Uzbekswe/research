import time

from orchestrator.graph import build_main_graph
from orchestrator.state import ResearchState


class ChiefEditorAgent:
    """Entry point for the multi-agent research system.

    Builds the main LangGraph workflow and runs the research task.
    Holds no research logic itself — only graph construction and execution.

    Modelled after GPT Researcher's ChiefEditorAgent / master.py.
    """

    def __init__(self, task: dict, output_dir: str = "./outputs") -> None:
        self.task = task
        self.output_dir = output_dir
        self.graph = build_main_graph(output_dir=output_dir)

    async def run_research_task(self) -> dict:
        """Run the full multi-agent research pipeline.

        Returns:
            Final ResearchState dict containing the completed report and all
            intermediate fields populated by individual agents.
        """
        print("\n" + "=" * 60)
        print("🚀 DEEP RESEARCHER — MULTI-AGENT MODE")
        print(f"   Query:      {self.task['query']}")
        print(f"   Sections:   {self.task.get('max_sections', 3)}")
        print(f"   Guidelines: {len(self.task.get('guidelines', []))} rules")
        print("=" * 60 + "\n")

        start_time = time.time()

        initial_state = ResearchState(
            task=self.task,
            initial_research="",
            sections=[],
            research_data=[],
            title="",
            headers={},
            date="",
            table_of_contents="",
            introduction="",
            conclusion="",
            sources=[],
            report="",
        )

        final_state = await self.graph.ainvoke(initial_state)

        elapsed = time.time() - start_time
        report = final_state.get("report", "")

        print("\n" + "=" * 60)
        print("✅ RESEARCH COMPLETE")
        print(f"   Title:         {final_state.get('title', 'N/A')}")
        print(f"   Sections:      {len(final_state.get('research_data', []))}")
        print(f"   Sources:       {len(final_state.get('sources', []))}")
        print(f"   Report length: {len(report)} chars")
        print(f"   Total time:    {elapsed:.1f}s")
        print("=" * 60)

        return final_state
