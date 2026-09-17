"""research-crew — a multi-agent research & report generation system.

A small crew of specialized CrewAI agents (Research Lead, Web Researcher,
Analyst, Report Writer) collaborate to research any topic or company and
produce a finished, sourced markdown report. Every agent hand-off is recorded
to a structured trace so the orchestration is visible, not a black box.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
