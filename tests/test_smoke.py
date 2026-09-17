"""Smoke tests that run without any API keys or network access.

These verify the parts we can exercise offline: model validation, config
secret-checking, the tracer's recording/serialization, and the search tool's
graceful handling of empty results and failures (with a fake client).

Tests that need CrewAI installed are skipped automatically if it is absent, so
`pytest` stays green in a minimal environment.
"""

from __future__ import annotations

import json

import pytest

from research_crew.config import Settings
from research_crew.models.schemas import (
    Depth,
    ResearchRequest,
    Source,
    TraceEvent,
)
from research_crew.tracing.tracer import Tracer


# --- models -----------------------------------------------------------------


def test_depth_tuning() -> None:
    assert Depth.QUICK.sub_question_count == 3
    assert Depth.DEEP.sub_question_count == 5
    assert Depth.DEEP.results_per_question >= Depth.QUICK.results_per_question


def test_request_trims_and_validates_topic() -> None:
    req = ResearchRequest(topic="  AI code assistants  ", depth=Depth.DEEP)
    assert req.topic == "AI code assistants"
    assert req.depth is Depth.DEEP


def test_request_rejects_blank_topic() -> None:
    with pytest.raises(ValueError):
        ResearchRequest(topic="   ")


def test_source_requires_valid_url() -> None:
    src = Source(url="https://example.com/article", title="Example")
    assert str(src.url).startswith("https://example.com")
    with pytest.raises(ValueError):
        Source(url="not-a-url")


# --- config -----------------------------------------------------------------


def test_require_secrets_reports_missing() -> None:
    settings = Settings(openai_api_key="", tavily_api_key="")
    with pytest.raises(RuntimeError) as exc:
        settings.require_secrets()
    assert "OPENAI_API_KEY" in str(exc.value)
    assert "TAVILY_API_KEY" in str(exc.value)


def test_require_secrets_passes_when_present() -> None:
    settings = Settings(openai_api_key="sk-x", tavily_api_key="tvly-x")
    settings.require_secrets()  # should not raise


def test_log_level_validation() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValueError):
        Settings(log_level="verbose")


# --- tracer -----------------------------------------------------------------


class _FakeTaskOutput:
    """Minimal stand-in for a CrewAI TaskOutput."""

    def __init__(self, agent: str, name: str, description: str, raw: str) -> None:
        self.agent = agent
        self.name = name
        self.description = description
        self.raw = raw


def test_tracer_records_and_serializes(tmp_path) -> None:
    tracer = Tracer(run_id="testrun", output_dir=tmp_path)
    tracer.record_task(
        _FakeTaskOutput("Research Lead", "plan", "Break down topic", "Q1\nQ2\nQ3")
    )
    tracer.record_task(
        _FakeTaskOutput("Web Researcher", "gather", "Search sources", "findings...")
    )

    events = tracer.events
    assert len(events) == 2
    assert all(isinstance(e, TraceEvent) for e in events)
    assert [e.sequence for e in events] == [0, 1]
    assert tracer.agents_used() == ["Research Lead", "Web Researcher"]

    path = tracer.write()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["event_count"] == 2
    assert data["agents_used"] == ["Research Lead", "Web Researcher"]
    assert len(data["events"]) == 2


# --- search tool (requires crewai) ------------------------------------------


class _FakeTavilyClient:
    """Fake Tavily client returning a scripted payload."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    def search(self, **_: object) -> dict:
        self.calls += 1
        return self._payload


def _make_tool(payload: dict):
    """Build a TavilySearchTool wired to a fake client (skips if no crewai)."""
    pytest.importorskip("crewai")
    from research_crew.tools.search_tool import SourceCollector, TavilySearchTool

    collector = SourceCollector()
    tool = TavilySearchTool(api_key="x", collector=collector, max_results=3)
    tool._client = _FakeTavilyClient(payload)  # inject fake client
    return tool, collector


def test_search_collects_sources() -> None:
    tool, collector = _make_tool(
        {
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "content": "alpha",
                    "score": 0.9,
                },
                {
                    "url": "https://example.com/b",
                    "title": "B",
                    "content": "beta",
                    "score": 0.8,
                },
            ]
        }
    )
    out = tool._run("what is alpha?")
    assert "https://example.com/a" in out
    assert "https://example.com/b" in out
    assert collector.count() == 2


def test_search_handles_empty_results() -> None:
    tool, collector = _make_tool({"results": []})
    out = tool._run("obscure query")
    assert "No web results" in out
    assert collector.count() == 0


def test_search_dedupes_by_url() -> None:
    tool, collector = _make_tool(
        {"results": [{"url": "https://x.com/1", "title": "X", "content": "c"}]}
    )
    tool._run("q1")
    tool._run("q2")
    assert collector.count() == 1
