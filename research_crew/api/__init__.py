"""FastAPI application exposing the research crew over HTTP."""

from __future__ import annotations

from research_crew.api.app import app, create_app

__all__ = ["app", "create_app"]
