"""Layer 4 graph structure tests — no real LLM or HTTP calls.

Tests verify that graphs compile, state TypedDicts have the right fields,
and the ChiefEditorAgent wires up correctly.
"""

import pytest

from orchestrator import ChiefEditorAgent
from orchestrator.graph import build_main_graph, build_section_subgraph
from orchestrator.state import DraftState, ResearchState


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_subgraph_builds():
    graph = build_section_subgraph()
    assert graph is not None
    assert hasattr(graph, "ainvoke")


def test_main_graph_builds():
    graph = build_main_graph()
    assert graph is not None
    assert hasattr(graph, "ainvoke")


def test_main_graph_with_custom_output_dir(tmp_path):
    graph = build_main_graph(output_dir=str(tmp_path))
    assert graph is not None


# ---------------------------------------------------------------------------
# State TypedDict structure
# ---------------------------------------------------------------------------

def test_research_state_keys():
    required_keys = [
        "task", "initial_research", "sections",
        "research_data", "title", "report",
    ]
    annotations = ResearchState.__annotations__
    for key in required_keys:
        assert key in annotations, f"ResearchState missing field: {key!r}"


def test_research_state_all_keys():
    expected = {
        "task", "initial_research", "sections", "research_data",
        "title", "headers", "date", "table_of_contents",
        "introduction", "conclusion", "sources", "report",
    }
    assert set(ResearchState.__annotations__) == expected


def test_draft_state_keys():
    required_keys = ["task", "topic", "draft", "review", "revision_notes"]
    annotations = DraftState.__annotations__
    for key in required_keys:
        assert key in annotations, f"DraftState missing field: {key!r}"


def test_draft_state_all_keys():
    expected = {"task", "topic", "draft", "review", "revision_notes", "guidelines"}
    assert set(DraftState.__annotations__) == expected


def test_draft_state_review_allows_none():
    annotation = DraftState.__annotations__["review"]
    # str | None → __args__ contains NoneType
    args = getattr(annotation, "__args__", ())
    assert type(None) in args, f"review should allow None, got: {annotation}"


# ---------------------------------------------------------------------------
# ChiefEditorAgent
# ---------------------------------------------------------------------------

def test_chief_editor_init():
    task = {"query": "test query", "max_sections": 2, "guidelines": []}
    agent = ChiefEditorAgent(task=task)
    assert agent.graph is not None
    assert agent.task["query"] == "test query"


def test_chief_editor_stores_output_dir(tmp_path):
    task = {"query": "test", "max_sections": 1, "guidelines": []}
    agent = ChiefEditorAgent(task=task, output_dir=str(tmp_path))
    assert agent.output_dir == str(tmp_path)


def test_chief_editor_run_research_task_is_async():
    import inspect
    task = {"query": "test", "max_sections": 1, "guidelines": []}
    agent = ChiefEditorAgent(task=task)
    assert inspect.iscoroutinefunction(agent.run_research_task)


# ---------------------------------------------------------------------------
# main.py load_task helper
# ---------------------------------------------------------------------------

def test_load_task_reads_json(tmp_path):
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent))  # ensure project root on path

    task_file = tmp_path / "task.json"
    task_file.write_text('{"query": "hello", "max_sections": 2}')

    # Import from the actual main module
    import importlib, types
    import main as main_mod
    task = main_mod.load_task(str(task_file))
    assert task["query"] == "hello"
    assert task["max_sections"] == 2


def test_load_task_exits_on_missing_file(tmp_path):
    import main as main_mod
    with pytest.raises(SystemExit):
        main_mod.load_task(str(tmp_path / "nonexistent.json"))
