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

import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from research_crew.config import get_settings
from research_crew.crew import ResearchCrew, ResearchError
from research_crew.logging_config import configure_logging
from research_crew.models.schemas import ResearchRequest, ResearchResponse

logger = logging.getLogger("research_crew.api")


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
