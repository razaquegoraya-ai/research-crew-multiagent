"""Centralized logging configuration.

Both the CLI and the API call :func:`configure_logging` once at startup so
console output — especially the agent hand-off trace — is consistently
formatted and readable.
"""

from __future__ import annotations

import logging

from research_crew.config import get_settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging idempotently.

    Args:
        level: Optional level override (e.g. ``"DEBUG"``). Defaults to the
            configured ``LOG_LEVEL`` setting.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # CrewAI/litellm are chatty at INFO; keep our own loggers front and center.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True
