"""FastAPI application: expose the research crew over HTTP.

Endpoints
---------
* ``GET  /health``  — liveness + configuration sanity check.
* ``POST /research`` — run the crew for a topic and return the report +
  metadata (agents used, time taken, sources count).

Because a crew run is CPU/IO-bound and blocking (it drives the LLM and web
searches synchronously), the endpoint executes it in a worker thread via
:func:`fastapi.concurrency.run_in_threadpool` so the event loop stays
responsive.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from research_crew.config import get_settings
from research_crew.crew import ResearchCrew, ResearchError
from research_crew.logging_config import configure_logging
from research_crew.models.schemas import ResearchRequest, ResearchResponse

logger = logging.getLogger("research_crew.api")

# Directory holding the single-page UI (research_crew/web/index.html).
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


class HealthResponse(BaseModel):
    """Response body for the health check."""

    status: str
    version: str
    model: str
    openai_key_configured: bool
    tavily_key_configured: bool


def create_app() -> FastAPI:
    """Application factory. Returns a configured :class:`FastAPI` instance."""
    from research_crew import __version__

    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="research-crew",
        version=__version__,
        summary="Multi-agent research & report generation crew.",
        description=(
            "A crew of specialized CrewAI agents researches a topic and "
            "produces a sourced markdown report, with a full orchestration "
            "trace saved alongside each run."
        ),
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the single-page dashboard UI."""
        index_file = _WEB_DIR / "index.html"
        if not index_file.exists():  # pragma: no cover - packaging guard
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="UI not found. Expected research_crew/web/index.html.",
            )
        return FileResponse(index_file, media_type="text/html")

    @app.get("/api/sample", tags=["ops"], include_in_schema=True)
    async def sample() -> JSONResponse:
        """Return the committed sample report + trace for a no-key demo.

        Lets the UI showcase a full run (report, metadata, trace timeline,
        sources) without any API keys or a live crew run. Sources are parsed
        from the report markdown so the count is real.
        """
        output_dir = Path(settings.output_dir)
        reports = sorted(output_dir.glob("*_sample.md"))
        traces = sorted(output_dir.glob("*_sample.json"))
        if not reports or not traces:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No committed sample files found in the output directory.",
            )

        report_md = reports[0].read_text(encoding="utf-8")
        trace_data = json.loads(traces[0].read_text(encoding="utf-8"))

        # Extract unique source URLs from the report body for the demo.
        seen: list[str] = []
        for url in _URL_RE.findall(report_md):
            cleaned = url.rstrip(".,")
            if cleaned not in seen:
                seen.append(cleaned)
        sources = [{"url": u, "title": "", "snippet": ""} for u in seen]

        return JSONResponse(
            {
                "report_markdown": report_md,
                "metadata": {
                    "topic": "The competitive landscape for AI code assistants in 2026",
                    "depth": "deep",
                    "model": settings.openai_model,
                    "agents_used": trace_data.get("agents_used", []),
                    "sub_questions_count": 5,
                    "sources_count": len(sources),
                    "elapsed_seconds": trace_data.get("elapsed_seconds", 0.0),
                    "trace_event_count": trace_data.get("event_count", 0),
                    "is_sample": True,
                },
                "sources": sources,
                "trace": trace_data.get("events", []),
            }
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Report liveness and whether required keys are configured.

        This never leaks secret values — only whether they are present.
        """
        return HealthResponse(
            status="ok",
            version=__version__,
            model=settings.openai_model,
            openai_key_configured=bool(settings.openai_api_key.strip()),
            tavily_key_configured=bool(settings.tavily_api_key.strip()),
        )

    @app.post(
        "/research",
        response_model=ResearchResponse,
        tags=["research"],
        status_code=status.HTTP_200_OK,
    )
    async def research(request: ResearchRequest) -> ResearchResponse:
        """Run the multi-agent research pipeline for a topic.

        Args:
            request: Topic and depth. Validated by Pydantic; malformed bodies
                yield a 422 automatically.

        Returns:
            The finished report, metadata and collected sources.

        Raises:
            HTTPException: 503 if required keys are missing; 500 on run failure.
        """
        logger.info(
            "POST /research topic=%r depth=%s", request.topic, request.depth.value
        )
        crew = ResearchCrew(settings=settings, verbose=True)
        try:
            return await run_in_threadpool(crew.run, request)
        except ResearchError as exc:
            message = str(exc)
            # Missing configuration is an operator problem, not a bad request.
            if "Missing required environment variable" in message:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=message
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message
            ) from exc
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            logger.exception("Unexpected error handling /research")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {exc}",
            ) from exc

    return app


# Module-level app for `uvicorn research_crew.api.app:app`.
app = create_app()
