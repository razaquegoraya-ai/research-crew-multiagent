"""Application configuration, loaded from environment / ``.env``.

All runtime configuration lives here and is read exactly once via
:func:`get_settings`. Secrets (API keys) are read from environment variables
and are *never* hardcoded. Missing required keys raise a clear error at the
moment they are actually needed rather than failing deep inside a request.
"""

from __future__ import annotations

import functools

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment.

    Values are read from process environment variables and, if present, a
    local ``.env`` file. Unknown environment variables are ignored so the
    process can coexist with unrelated configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (required to actually run the crew) ---
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key used by every agent's LLM.",
    )
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key used by the web search tool.",
    )

    # --- LLM configuration ---
    openai_model: str = Field(
        default="gpt-4o",
        description="Chat completion model the agents run on.",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for the agents.",
    )

    # --- Search configuration ---
    search_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum Tavily results fetched per sub-question.",
    )

    # --- I/O & logging ---
    output_dir: str = Field(
        default="outputs",
        description="Directory where reports and traces are written.",
    )
    log_level: str = Field(default="INFO", description="Root logging level.")

    # --- API server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Uppercase and validate the log level string."""
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    def require_secrets(self) -> None:
        """Ensure the secrets needed to run the crew are present.

        Raises:
            RuntimeError: If ``OPENAI_API_KEY`` or ``TAVILY_API_KEY`` is unset.
                The message names the missing variable(s) so the operator can
                fix configuration without reading a stack trace.
        """
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                ("TAVILY_API_KEY", self.tavily_api_key),
            )
            if not value.strip()
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    The result is cached so the ``.env`` file and environment are parsed only
    once per process. Tests can clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
