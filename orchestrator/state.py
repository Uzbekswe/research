import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """Shared state for the main research graph.

    Every agent reads from and writes back to this dict.
    No agent calls another agent directly — only state is passed.
    """

    task: dict                  # full task.json config dict
    initial_research: str       # broad context from first browser sweep
    sections: list[str]         # planned report sections from editor
    research_data: list[dict]   # completed section drafts: [{"section": str, "draft": str, "sources": list}]
    title: str
    headers: dict               # {"introduction": str, "table_of_contents": str, ...}
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: list[str]
    report: str                 # final assembled report


class DraftState(TypedDict):
    """Isolated state for the per-section sub-graph.

    One DraftState instance per section — running concurrently.
    Isolation prevents race conditions between parallel sections.
    """

    task: dict              # inherited from ResearchState
    topic: str              # the section title being researched
    draft: str              # current draft of this section
    review: str | None      # None = approved, string = revision notes
    revision_notes: str     # accumulated history of all revision notes
    guidelines: list[str]   # quality criteria the reviewer checks against
