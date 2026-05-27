"""Layer 1 smoke tests — instantiation and interface checks.

These tests verify that DeepResearcher can be constructed correctly and that
all public methods exist with the right signatures.  No real LLM or HTTP
calls are made; tests rely entirely on object construction and attribute
inspection.
"""

import inspect
import pytest

from researcher import DeepResearcher
from researcher.config import Config
from researcher.memory.research_memory import ResearchMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def researcher() -> DeepResearcher:
    """A DeepResearcher instance with a simple query and default config."""
    return DeepResearcher(query="What is the history of the Python programming language?")


@pytest.fixture
def researcher_with_options() -> DeepResearcher:
    """A DeepResearcher instance exercising all constructor parameters."""
    return DeepResearcher(
        query="Explain quantum entanglement",
        report_type="outline_report",
        report_source="web",
        source_urls=["https://example.com/quantum"],
        document_urls=["https://example.com/doc.pdf"],
        config_path=None,
        websocket=None,
        agent="science_agent",
        role="You are a physics researcher.",
        parent_query="Physics concepts",
        subtopic="Quantum mechanics",
        headers={"X-Custom": "header"},
        max_subtopics=5,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_instantiation(self, researcher):
        assert researcher is not None

    def test_query_stored(self, researcher):
        assert researcher.query == "What is the history of the Python programming language?"

    def test_defaults(self, researcher):
        assert researcher.report_type == "research_report"
        assert researcher.report_source == "web"
        assert researcher.source_urls == []
        assert researcher.document_urls == []
        assert researcher.parent_query == ""
        assert researcher.subtopic == ""
        assert researcher.headers == {}
        assert researcher.max_subtopics == 3
        assert researcher.verbose is True
        assert researcher.context == ""
        assert researcher.research_costs == 0.0

    def test_config_created(self, researcher):
        assert isinstance(researcher.cfg, Config)

    def test_memory_created(self, researcher):
        assert isinstance(researcher.memory, ResearchMemory)

    def test_role_auto_set(self, researcher):
        """Role should be populated from get_agent_role_prompt if not supplied."""
        assert isinstance(researcher.role, str)
        assert len(researcher.role) > 10

    def test_custom_role_preserved(self, researcher_with_options):
        assert researcher_with_options.role == "You are a physics researcher."

    def test_all_options_stored(self, researcher_with_options):
        r = researcher_with_options
        assert r.report_type == "outline_report"
        assert r.source_urls == ["https://example.com/quantum"]
        assert r.document_urls == ["https://example.com/doc.pdf"]
        assert r.agent == "science_agent"
        assert r.parent_query == "Physics concepts"
        assert r.subtopic == "Quantum mechanics"
        assert r.headers == {"X-Custom": "header"}
        assert r.max_subtopics == 5

    def test_config_defaults_loaded(self, researcher):
        cfg = researcher.cfg
        assert cfg.FAST_LLM == "openai:gpt-4o-mini"
        assert cfg.SMART_LLM == "openai:gpt-4o"
        assert cfg.RETRIEVER == "duckduckgo"
        assert cfg.SCRAPER == "bs4"
        assert cfg.TEMPERATURE == 0.4
        assert cfg.REPORT_FORMAT == "APA"
        assert cfg.TOTAL_WORDS == 1200
        assert cfg.MAX_ITERATIONS == 3
        assert cfg.MAX_SUBTOPICS == 3


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Verify every method in the public API exists and has the right signature."""

    _ASYNC_METHODS = [
        "conduct_research",
        "write_report",
        "write_report_conclusion",
        "get_subtopics",
    ]

    _SYNC_GETTERS = [
        "get_research_context",
        "get_source_urls",
        "get_research_sources",
        "get_costs",
        "get_research_images",
    ]

    _SYNC_SETTERS = [
        "set_verbose",
        "add_costs",
    ]

    def test_async_methods_exist(self, researcher):
        for name in self._ASYNC_METHODS:
            method = getattr(researcher, name, None)
            assert method is not None, f"Missing method: {name}"
            assert inspect.iscoroutinefunction(method), f"{name} should be async"

    def test_sync_getters_exist(self, researcher):
        for name in self._SYNC_GETTERS:
            method = getattr(researcher, name, None)
            assert method is not None, f"Missing getter: {name}"
            assert callable(method), f"{name} should be callable"
            assert not inspect.iscoroutinefunction(method), f"{name} should be sync"

    def test_sync_setters_exist(self, researcher):
        for name in self._SYNC_SETTERS:
            method = getattr(researcher, name, None)
            assert method is not None, f"Missing setter: {name}"
            assert callable(method), f"{name} should be callable"


# ---------------------------------------------------------------------------
# Getter / setter behaviour (no network calls)
# ---------------------------------------------------------------------------

class TestGettersSetters:
    def test_get_research_context_initial(self, researcher):
        assert researcher.get_research_context() == ""

    def test_get_source_urls_initial(self, researcher):
        assert researcher.get_source_urls() == []

    def test_get_research_sources_initial(self, researcher):
        assert researcher.get_research_sources() == []

    def test_get_costs_initial(self, researcher):
        assert researcher.get_costs() == 0.0

    def test_get_research_images_initial(self, researcher):
        assert researcher.get_research_images() == []

    def test_set_verbose(self, researcher):
        researcher.set_verbose(False)
        assert researcher.verbose is False
        researcher.set_verbose(True)
        assert researcher.verbose is True

    def test_add_costs_float(self, researcher):
        researcher.add_costs(0.005)
        researcher.add_costs(0.003)
        assert abs(researcher.get_costs() - 0.008) < 1e-9
        assert abs(researcher.research_costs - 0.008) < 1e-9

    def test_add_costs_dict(self, researcher):
        """Dict form is produced by action-layer cost callbacks."""
        researcher.add_costs({"llm": "openai:gpt-4o", "tokens": 0, "cost": 0.002})
        assert abs(researcher.get_costs() - 0.002) < 1e-9

    def test_add_costs_dict_no_cost_key(self, researcher):
        """Dict without 'cost' key should add 0.0 (not crash)."""
        researcher.add_costs({"llm": "openai:gpt-4o", "tokens": 500})
        assert researcher.get_costs() == 0.0

    def test_add_costs_invalid_type_ignored(self, researcher):
        """Non-numeric, non-dict value should be silently ignored."""
        researcher.add_costs("not a number")
        assert researcher.get_costs() == 0.0


# ---------------------------------------------------------------------------
# Top-level import smoke test
# ---------------------------------------------------------------------------

def test_package_exports_deep_researcher():
    from researcher import DeepResearcher as DR
    assert DR is DeepResearcher


@pytest.mark.asyncio
async def test_write_report_returns_empty_without_context(researcher, monkeypatch):
    """write_report should not crash when context is empty; returns '' on LLM error."""
    # Patch the underlying action to avoid any real LLM call.
    async def fake_write(*args, **kwargs):
        return "Fake report content"

    import researcher.agent as agent_mod
    monkeypatch.setattr(agent_mod, "_write_report", fake_write)

    result = await researcher.write_report()
    assert isinstance(result, str)
