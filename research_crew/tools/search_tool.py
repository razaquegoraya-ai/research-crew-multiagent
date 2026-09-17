"""A Tavily-backed web search tool for the Web Researcher agent.

Design notes
------------
* **Sources are collected, not inferred.** Every result the tool returns is
  also recorded in a shared :class:`SourceCollector`. The report's
  ``sources_count`` therefore reflects real retrievals, independent of whatever
  the LLM chooses to write — a much stronger signal for a portfolio demo.
* **Failures are graceful.** Rate limits are retried with exponential backoff;
  empty results return a clear, model-readable message rather than raising;
  hard failures return an error string the agent can reason about instead of
  crashing the whole run.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from research_crew.models.schemas import Source

logger = logging.getLogger("research_crew.tools.search")

_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 2.0


class SourceCollector:
    """Thread-safe registry of unique sources gathered across the run.

    De-duplicates by URL so the same page surfacing under two sub-questions is
    counted once. Shared between the tool and the crew so metadata can report
    an accurate source count.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_url: dict[str, Source] = {}

    def add(self, source: Source) -> None:
        """Add a source, keeping the first-seen entry for a given URL."""
        key = str(source.url)
        with self._lock:
            self._by_url.setdefault(key, source)

    def all(self) -> list[Source]:
        """Return all unique sources in insertion order."""
        with self._lock:
            return list(self._by_url.values())

    def count(self) -> int:
        """Number of unique sources collected so far."""
        with self._lock:
            return len(self._by_url)


class _SearchInput(BaseModel):
    """Arguments accepted by :class:`TavilySearchTool`."""

    query: str = Field(
        ...,
        description="A focused search query, ideally a single sub-question.",
    )


class TavilySearchTool(BaseTool):
    """Search the live web via Tavily and return sourced findings.

    The tool returns a compact, model-friendly string: each result as a
    numbered block with its title, URL, and a content snippet. URLs are always
    included so downstream agents can cite them.
    """

    name: str = "web_search"
    description: str = (
        "Search the live web for up-to-date information. Input is a single "
        "focused query or sub-question. Returns a numbered list of results, "
        "each with a title, source URL, and a content snippet. Always cite the "
        "returned URLs in your findings."
    )
    args_schema: type[BaseModel] = _SearchInput

    # Private (non-serialized) runtime dependencies.
    _api_key: str = PrivateAttr()
    _max_results: int = PrivateAttr()
    _collector: SourceCollector = PrivateAttr()
    _client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        api_key: str,
        collector: SourceCollector,
        max_results: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize the tool.

        Args:
            api_key: Tavily API key.
            collector: Shared collector that accumulates unique sources.
            max_results: Maximum results to request per query.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._max_results = max_results
        self._collector = collector

    def _get_client(self) -> Any:
        """Lazily construct and cache the Tavily client.

        Raises:
            RuntimeError: If the ``tavily-python`` package is not installed.
        """
        if self._client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:  # pragma: no cover - import-time guard
                raise RuntimeError(
                    "tavily-python is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc
            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def _run(self, query: str) -> str:
        """Execute a search and return formatted, sourced results.

        This never raises for expected operational problems (empty results,
        transient rate limits, provider errors); it returns a descriptive
        string so the agent can adapt. Unexpected programming errors still
        propagate.

        Args:
            query: The search query / sub-question.

        Returns:
            A numbered, human- and model-readable results block, or a clear
            message describing why no usable results were returned.
        """
        query = (query or "").strip()
        if not query:
            return "No query was provided. Supply a focused sub-question to search."

        try:
            response = self._search_with_retries(query)
        except Exception as exc:  # noqa: BLE001 - surface as text, keep run alive
            logger.warning("Search failed for %r: %s", query, exc)
            return (
                f"Web search for '{query}' failed after retries: {exc}. "
                "Proceed using other sub-questions and note the gap."
            )

        results: list[dict[str, Any]] = response.get("results", []) or []
        if not results:
            logger.info("No results for query %r", query)
            return (
                f"No web results were found for '{query}'. "
                "Try a broader or rephrased query, or note this as a gap."
            )

        lines: list[str] = [f"Search results for: {query}", ""]
        for index, item in enumerate(results, start=1):
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "Untitled").strip()
            content = (item.get("content") or "").strip()
            score = item.get("score")

            if url:
                self._record_source(url, title, content, query, score)

            snippet = content[:500] + ("…" if len(content) > 500 else "")
            lines.append(f"[{index}] {title}")
            lines.append(f"    URL: {url or 'n/a'}")
            if snippet:
                lines.append(f"    Snippet: {snippet}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _record_source(
        self,
        url: str,
        title: str,
        content: str,
        query: str,
        score: Any,
    ) -> None:
        """Add a validated source to the collector, ignoring bad URLs."""
        try:
            self._collector.add(
                Source(
                    url=url,
                    title=title,
                    snippet=content[:300],
                    query=query,
                    score=float(score) if isinstance(score, (int, float)) else None,
                )
            )
        except Exception:  # noqa: BLE001 - a malformed URL shouldn't stop search
            logger.debug("Skipped un-parseable source URL: %r", url)

    def _search_with_retries(self, query: str) -> dict[str, Any]:
        """Call Tavily with exponential backoff on transient failures.

        Args:
            query: The search query.

        Returns:
            The raw Tavily response dictionary.

        Raises:
            Exception: The last error encountered if all retries are exhausted.
        """
        client = self._get_client()
        backoff = _INITIAL_BACKOFF_SECONDS
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return client.search(
                    query=query,
                    max_results=self._max_results,
                    search_depth="advanced",
                )
            except Exception as exc:  # noqa: BLE001 - decide retry vs re-raise
                last_error = exc
                if not self._is_retryable(exc) or attempt == _MAX_RETRIES:
                    raise
                logger.info(
                    "Transient search error (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        # Unreachable, but keeps type-checkers happy.
        raise last_error if last_error else RuntimeError("search failed")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Heuristically decide whether an error is worth retrying."""
        text = str(exc).lower()
        markers = ("rate limit", "429", "timeout", "timed out", "temporarily", "503", "502")
        return any(marker in text for marker in markers)
