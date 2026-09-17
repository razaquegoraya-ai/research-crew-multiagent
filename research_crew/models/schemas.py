"""Structured data models shared across the crew, tracer and API.

These Pydantic models are the contract between layers: the API validates
incoming requests against :class:`ResearchRequest`, the tracer emits
:class:`TraceEvent` records, and the crew returns a :class:`ResearchResponse`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Depth(str, Enum):
    """How thorough a research run should be.

    ``QUICK`` favors speed (fewer sub-questions and sources); ``DEEP`` favors
    coverage. The value drives sub-question count and per-question search depth.
    """

    QUICK = "quick"
    DEEP = "deep"

    @property
    def sub_question_count(self) -> int:
        """Target number of sub-questions for this depth."""
        return 3 if self is Depth.QUICK else 5

    @property
    def results_per_question(self) -> int:
        """Suggested number of search results per sub-question."""
        return 4 if self is Depth.QUICK else 6


class ResearchRequest(BaseModel):
    """A request to research a topic and produce a report."""

    topic: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="The topic or company to research.",
        examples=["The competitive landscape for AI code assistants in 2026"],
    )
    depth: Depth = Field(
        default=Depth.QUICK,
        description="Research depth: 'quick' or 'deep'.",
    )

    @field_validator("topic")
    @classmethod
    def _strip_topic(cls, value: str) -> str:
        """Trim surrounding whitespace and reject blank topics."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("topic must not be blank")
        return cleaned


class Source(BaseModel):
    """A single web source gathered during research."""

    url: HttpUrl = Field(..., description="Canonical URL of the source.")
    title: str = Field(default="", description="Page or article title.")
    snippet: str = Field(
        default="",
        description="Short excerpt / summary returned by the search provider.",
    )
    query: str = Field(
        default="",
        description="The sub-question / query that surfaced this source.",
    )
    score: float | None = Field(
        default=None,
        description="Relevance score from the search provider, if available.",
    )


class TraceEvent(BaseModel):
    """One recorded event in the multi-agent orchestration trace.

    A trace is an ordered list of these. Each event captures which agent ran,
    what it received, what it produced, and how long it took — the visible
    proof that real hand-offs are happening.
    """

    sequence: int = Field(..., ge=0, description="Monotonic event index.")
    event_type: str = Field(
        ...,
        description="Kind of event, e.g. 'task_completed' or 'tool_call'.",
    )
    agent: str = Field(..., description="Role name of the acting agent.")
    task: str = Field(
        default="",
        description="Task name / short label for the unit of work.",
    )
    input: str = Field(default="", description="Input handed to the agent.")
    output: str = Field(default="", description="Output the agent produced.")
    started_at: datetime = Field(..., description="Event start (UTC).")
    ended_at: datetime = Field(..., description="Event end (UTC).")
    duration_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock duration of the event."
    )


class ReportMetadata(BaseModel):
    """Metadata describing how a report was produced."""

    topic: str
    depth: Depth
    model: str = Field(..., description="LLM model the crew ran on.")
    agents_used: list[str] = Field(
        default_factory=list, description="Distinct agent roles that ran."
    )
    sub_questions_count: int = Field(default=0, ge=0)
    sources_count: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    generated_at: datetime = Field(..., description="Completion time (UTC).")
    report_file: str = Field(default="", description="Path to the saved report.")
    trace_file: str = Field(default="", description="Path to the saved trace.")
    trace_event_count: int = Field(default=0, ge=0)


class ResearchResponse(BaseModel):
    """The full result returned to an API caller or CLI user."""

    report_markdown: str = Field(..., description="The finished markdown report.")
    metadata: ReportMetadata
    sources: list[Source] = Field(
        default_factory=list, description="De-duplicated sources cited."
    )
    trace: list[TraceEvent] = Field(
        default_factory=list,
        description="Ordered orchestration trace (agent hand-offs & tool calls).",
    )
