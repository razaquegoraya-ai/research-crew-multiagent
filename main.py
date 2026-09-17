"""Command-line entry point for research-crew.

Examples:
    python main.py --topic "Anthropic's position in the AI market" --depth deep
    python main.py -t "State of RAG frameworks in 2026"

Run ``python main.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import sys

from research_crew.crew import ResearchCrew, ResearchError
from research_crew.logging_config import configure_logging
from research_crew.models.schemas import Depth, ResearchRequest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="research-crew",
        description="Run a multi-agent research crew and produce a sourced report.",
    )
    parser.add_argument(
        "-t",
        "--topic",
        required=True,
        help="The topic or company to research.",
    )
    parser.add_argument(
        "-d",
        "--depth",
        choices=[d.value for d in Depth],
        default=Depth.QUICK.value,
        help="Research depth (default: quick).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce agent narration (hand-off trace is still logged).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI main. Returns a process exit code."""
    args = _parse_args(argv)
    configure_logging()

    request = ResearchRequest(topic=args.topic, depth=Depth(args.depth))
    crew = ResearchCrew(verbose=not args.quiet)

    try:
        response = crew.run(request)
    except ResearchError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    meta = response.metadata
    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)
    print(f"Topic          : {meta.topic}")
    print(f"Depth          : {meta.depth.value}")
    print(f"Model          : {meta.model}")
    print(f"Agents used    : {', '.join(meta.agents_used) or 'n/a'}")
    print(f"Sub-questions  : {meta.sub_questions_count}")
    print(f"Sources        : {meta.sources_count}")
    print(f"Trace events   : {meta.trace_event_count}")
    print(f"Elapsed        : {meta.elapsed_seconds:.1f}s")
    print(f"Report file    : {meta.report_file}")
    print(f"Trace file     : {meta.trace_file}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
