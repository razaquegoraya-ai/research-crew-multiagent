"""The orchestration tracer — a first-class, visible record of hand-offs.

The whole point of a multi-agent system is that distinct agents actually pass
work between each other. That is invisible unless you show it, so this module
treats the trace as a product feature:

* Every agent hand-off (task completion) and every tool call is recorded as a
  :class:`~research_crew.models.schemas.TraceEvent`.
* Each event is logged to the console in real time with clear framing, so a
  live run reads like a relay of specialists.
* The full ordered trace is written to ``outputs/trace_<timestamp>.json`` so
  it can be replayed, diffed, or embedded in a portfolio demo.

The tracer is thread-safe because CrewAI may invoke callbacks from worker
threads (e.g. when the crew runs inside a FastAPI thread pool).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_crew.models.schemas import TraceEvent

logger = logging.getLogger("research_crew.trace")

# Maximum characters of any single input/output stored per event. Traces are a
# demo artifact, not a data lake — keep them readable and bounded.
_MAX_FIELD_CHARS = 4000


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    """Truncate ``text`` to ``limit`` chars with an explicit marker."""
    if len(text) <= limit:
        return text
    remaining = len(text) - limit
    return f"{text[:limit]}… [truncated {remaining} chars]"


def _stringify(value: Any) -> str:
    """Best-effort convert an arbitrary CrewAI callback payload to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # CrewAI output objects expose the rendered text on `.raw`.
    for attr in ("raw", "output", "result", "summary"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return str(value)


class Tracer:
    """Collects, logs, and persists the orchestration trace for one run.

    Args:
        run_id: Stable identifier for this run (used in the trace filename).
        output_dir: Directory the trace JSON is written to.
    """

    def __init__(self, run_id: str, output_dir: str | Path = "outputs") -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._run_started_at = _utcnow()
        # Marks the end of the previous hand-off; the next task's duration is
        # measured from here, so per-agent timing reflects real elapsed work.
        self._last_marker = self._run_started_at

    # -- recording -----------------------------------------------------------

    def _next_sequence(self) -> int:
        """Return the next monotonic sequence number (caller holds the lock)."""
        value = self._sequence
        self._sequence += 1
        return value

    def record_task(self, task_output: Any) -> None:
        """Record a completed task hand-off.

        Wired to CrewAI's ``task_callback``. Captures the acting agent, the
        task description (its input) and the produced output, timing the work
        from the previous hand-off marker to now.

        Args:
            task_output: The CrewAI ``TaskOutput`` for the completed task.
        """
        ended = _utcnow()
        agent = _stringify(getattr(task_output, "agent", "") or "unknown-agent")
        task_name = _stringify(
            getattr(task_output, "name", "") or getattr(task_output, "description", "")
        )
        description = _stringify(getattr(task_output, "description", ""))
        output_text = _stringify(getattr(task_output, "raw", task_output))

        with self._lock:
            started = self._last_marker
            event = TraceEvent(
                sequence=self._next_sequence(),
                event_type="task_completed",
                agent=agent.strip() or "unknown-agent",
                task=_truncate(task_name, 200),
                input=_truncate(description),
                output=_truncate(output_text),
                started_at=started,
                ended_at=ended,
                duration_seconds=max((ended - started).total_seconds(), 0.0),
            )
            self._events.append(event)
            self._last_marker = ended

        self._log_event(event)

    def record_step(self, step_output: Any) -> None:
        """Record an intra-task agent step (often a tool call).

        Wired to CrewAI's ``step_callback``. These are finer-grained than task
        hand-offs and make tool usage (e.g. web searches) visible in the trace.
        Failures here never interrupt the run.

        Args:
            step_output: The CrewAI step/agent-action payload.
        """
        try:
            ended = _utcnow()
            tool = getattr(step_output, "tool", None)
            tool_input = getattr(step_output, "tool_input", None)
            text = _stringify(getattr(step_output, "text", step_output))

            event_type = "tool_call" if tool else "agent_step"
            input_text = ""
            if tool:
                input_text = f"{tool}({_stringify(tool_input)})"

            with self._lock:
                started = self._last_marker
                event = TraceEvent(
                    sequence=self._next_sequence(),
                    event_type=event_type,
                    agent="(step)",
                    task=_truncate(_stringify(tool) or "step", 200),
                    input=_truncate(input_text or text),
                    output=_truncate(text),
                    started_at=started,
                    ended_at=ended,
                    duration_seconds=max((ended - started).total_seconds(), 0.0),
                )
                self._events.append(event)
                # Steps do not advance the hand-off marker; task boundaries do.
            self._log_event(event, step=True)
        except Exception:  # pragma: no cover - tracing must never break a run
            logger.debug("Failed to record step event", exc_info=True)

    # -- console output ------------------------------------------------------

    def _log_event(self, event: TraceEvent, *, step: bool = False) -> None:
        """Emit a single, readable console line for an event."""
        if step:
            logger.info(
                "  ↳ [%02d] %s: %s (%.2fs)",
                event.sequence,
                event.event_type,
                event.task,
                event.duration_seconds,
            )
            return
        logger.info(
            "──▶ [%02d] HAND-OFF → %s | %s | %.2fs",
            event.sequence,
            event.agent,
            event.task or event.event_type,
            event.duration_seconds,
        )

    # -- persistence ---------------------------------------------------------

    @property
    def events(self) -> list[TraceEvent]:
        """A copy of the recorded events, in order."""
        with self._lock:
            return list(self._events)

    def elapsed_seconds(self) -> float:
        """Total wall-clock seconds since the tracer was created."""
        return max((_utcnow() - self._run_started_at).total_seconds(), 0.0)

    def agents_used(self) -> list[str]:
        """Distinct agent roles seen in task hand-offs, in first-seen order."""
        seen: list[str] = []
        for event in self.events:
            if event.event_type == "task_completed" and event.agent not in seen:
                seen.append(event.agent)
        return seen

    def write(self) -> Path:
        """Serialize the trace to ``outputs/trace_<run_id>.json``.

        Returns:
            The path the trace was written to.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"trace_{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "run_started_at": self._run_started_at.isoformat(),
            "run_ended_at": _utcnow().isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "agents_used": self.agents_used(),
            "event_count": len(self._events),
            "events": [
                json.loads(event.model_dump_json()) for event in self.events
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Trace written to %s (%d events)", path, len(self._events))
        return path
