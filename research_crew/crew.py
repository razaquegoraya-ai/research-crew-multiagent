"""Assemble and run the research crew end-to-end.

:class:`ResearchCrew` is the single entry point used by both the CLI and the
API. It wires the LLM, the search tool, the four agents, the task graph and the
tracer together, runs the crew, persists the report and trace to ``outputs/``,
and returns a structured :class:`~research_crew.models.schemas.ResearchResponse`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from crewai import LLM, Crew, Process

from research_crew.agents.crew_agents import build_agents
from research_crew.config import Settings, get_settings
from research_crew.models.schemas import (
    Depth,
    ReportMetadata,
    ResearchRequest,
    ResearchResponse,
)
from research_crew.tasks.crew_tasks import build_tasks
from research_crew.tools.search_tool import SourceCollector, TavilySearchTool
from research_crew.tracing.tracer import Tracer

logger = logging.getLogger("research_crew.crew")


class ResearchError(RuntimeError):
    """Raised when a research run fails in a way the caller should handle."""


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn a topic into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "report"


class ResearchCrew:
    """Runs a single research topic through the multi-agent pipeline.

    Args:
        settings: Optional settings override. Defaults to the process settings.
        verbose: Whether agents narrate their reasoning to the console.
    """

    def __init__(self, settings: Settings | None = None, *, verbose: bool = True) -> None:
        self.settings = settings or get_settings()
        self.verbose = verbose

    def run(self, request: ResearchRequest) -> ResearchResponse:
        """Execute the full research pipeline for one request.

        Args:
            request: The validated research request.

        Returns:
            The finished report, its metadata, and the collected sources.

        Raises:
            ResearchError: If required secrets are missing or the crew fails.
        """
        try:
            self.settings.require_secrets()
        except RuntimeError as exc:
            raise ResearchError(str(exc)) from exc

        topic, depth = request.topic, request.depth
        run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{_slugify(topic)}"
        output_dir = Path(self.settings.output_dir)

        logger.info("Starting research run '%s' (depth=%s)", run_id, depth.value)
        tracer = Tracer(run_id=run_id, output_dir=output_dir)
        collector = SourceCollector()

        crew = self._assemble_crew(topic, depth, tracer, collector)

        try:
            result = crew.kickoff()
        except Exception as exc:  # noqa: BLE001 - normalize into ResearchError
            # Still persist whatever trace we captured before failing.
            tracer.write()
            logger.exception("Crew execution failed for run '%s'", run_id)
            raise ResearchError(f"Crew execution failed: {exc}") from exc

        report_markdown = self._extract_report(result)
        report_path = self._write_report(output_dir, run_id, report_markdown)
        trace_path = tracer.write()

        metadata = ReportMetadata(
            topic=topic,
            depth=depth,
            model=self.settings.openai_model,
            agents_used=tracer.agents_used(),
            sub_questions_count=depth.sub_question_count,
            sources_count=collector.count(),
            elapsed_seconds=round(tracer.elapsed_seconds(), 2),
            generated_at=datetime.now(timezone.utc),
            report_file=str(report_path),
            trace_file=str(trace_path),
            trace_event_count=len(tracer.events),
        )

        logger.info(
            "Run '%s' complete: %d sources, %d trace events, %.1fs",
            run_id,
            metadata.sources_count,
            metadata.trace_event_count,
            metadata.elapsed_seconds,
        )

        return ResearchResponse(
            report_markdown=report_markdown,
            metadata=metadata,
            sources=collector.all(),
        )

    # -- internals -----------------------------------------------------------

    def _assemble_crew(
        self,
        topic: str,
        depth: Depth,
        tracer: Tracer,
        collector: SourceCollector,
    ) -> Crew:
        """Build the LLM, tool, agents, tasks and Crew for one run."""
        llm = LLM(
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            temperature=self.settings.llm_temperature,
        )
        search_tool = TavilySearchTool(
            api_key=self.settings.tavily_api_key,
            collector=collector,
            max_results=self.settings.search_max_results,
        )
        agents = build_agents(llm, search_tool, verbose=self.verbose)
        tasks = build_tasks(agents, topic, depth)

        return Crew(
            agents=[
                agents.research_lead,
                agents.web_researcher,
                agents.analyst,
                agents.report_writer,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=self.verbose,
            # ⭐ Tracing hooks — the visible proof of orchestration.
            task_callback=tracer.record_task,
            step_callback=tracer.record_step,
        )

    @staticmethod
    def _extract_report(result: object) -> str:
        """Extract the final markdown string from a CrewAI kickoff result."""
        for attr in ("raw", "output"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        text = str(result).strip()
        if not text:
            raise ResearchError("Crew produced an empty report.")
        return text

    def _write_report(self, output_dir: Path, run_id: str, markdown: str) -> Path:
        """Write the markdown report to a timestamped file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"report_{run_id}.md"
        path.write_text(markdown, encoding="utf-8")
        logger.info("Report written to %s", path)
        return path
