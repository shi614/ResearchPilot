"""Tavily web-search and page-extraction tools.

`WebSearchTools.tools` exposes the LangChain tools for LLM tool calling
(`bind_tools`), while `search()` / `extract()` execute them with retries,
error mapping and result normalisation. The research agent uses both: the LLM
decides *which* tool calls to make, this module makes them safely.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain_core.tools import BaseTool, ToolException
from langchain_tavily import TavilyExtract, TavilySearch

from app.config import Settings
from app.exceptions import ConfigurationError, ExternalServiceError, RateLimitError
from app.models.domain import WebPageExtract, WebSearchResult
from app.utils.retry import call_with_retry

logger = logging.getLogger(__name__)

SERVICE = "Tavily"
MAX_QUERY_CHARS = 400  # Tavily rejects longer queries
MAX_EXTRACT_URLS = 5
MAX_EXTRACT_CHARS = 8000  # keep extracted pages small to save LLM tokens
_STATUS_PATTERN = re.compile(r"Error (\d{3})")
# 432/433 are Tavily's plan / pay-as-you-go credit limit codes.
_RATE_LIMIT_CODES = {429, 432, 433}


class _TavilyCallError(Exception):
    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def _raise_if_error(raw: Any) -> dict[str, Any]:
    """langchain-tavily returns `{"error": exc}` instead of raising; undo that."""
    if not isinstance(raw, dict):
        raise _TavilyCallError(None, f"unexpected response type {type(raw).__name__}")
    if "error" in raw:
        detail = str(raw["error"])
        match = _STATUS_PATTERN.search(detail)
        raise _TavilyCallError(int(match.group(1)) if match else None, detail)
    return raw


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, _TavilyCallError) and (
        exc.status is None or exc.status in _RATE_LIMIT_CODES or exc.status >= 500
    )


def _to_service_error(exc: _TavilyCallError) -> ExternalServiceError:
    if exc.status in (401, 403):
        return ExternalServiceError(SERVICE, "the API key was rejected. Check TAVILY_API_KEY.")
    if exc.status in _RATE_LIMIT_CODES:
        return RateLimitError(SERVICE, "rate limit or monthly credit limit reached.")
    if exc.status == 400:
        return ExternalServiceError(SERVICE, "the request was rejected as invalid.")
    if exc.status is not None and exc.status >= 500:
        return ExternalServiceError(SERVICE, f"service error ({exc.status}). Try again later.")
    return ExternalServiceError(SERVICE, "request failed (network error or unexpected response).")


class WebSearchTools:
    def __init__(
        self,
        search_tool: BaseTool,
        extract_tool: BaseTool,
        *,
        max_attempts: int = 3,
        sleep: Any = None,
    ) -> None:
        self.search_tool = search_tool
        self.extract_tool = extract_tool
        self._retry_kwargs: dict[str, Any] = {"max_attempts": max_attempts}
        if sleep is not None:
            self._retry_kwargs["sleep"] = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> WebSearchTools:
        if settings.tavily_api_key is None:
            raise ConfigurationError("TAVILY_API_KEY is not set; web research is unavailable.")
        api_key = settings.tavily_api_key.get_secret_value()
        search = TavilySearch(
            tavily_api_key=api_key,
            max_results=settings.tavily_max_results,
            search_depth=settings.tavily_search_depth,
            topic="general",
            include_answer=False,  # we synthesise answers ourselves; keep raw evidence
            include_raw_content=False,
            include_images=False,
        )
        extract = TavilyExtract(tavily_api_key=api_key, extract_depth="basic")
        return cls(search, extract)

    @property
    def tools(self) -> list[BaseTool]:
        """LangChain tools to bind to the LLM for tool calling."""
        return [self.search_tool, self.extract_tool]

    def search(
        self,
        query: str,
        *,
        time_range: Literal["day", "week", "month", "year"] | None = None,
        include_domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        """Run one web search. Returns [] when nothing is found."""
        query = " ".join(query.split())[:MAX_QUERY_CHARS]
        if not query:
            raise ValueError("Search query must not be empty")
        payload: dict[str, Any] = {"query": query}
        if time_range:
            payload["time_range"] = time_range
        if include_domains:
            payload["include_domains"] = include_domains

        try:
            raw = self._invoke(self.search_tool, payload, f"Tavily search '{query[:60]}'")
        except ToolException:
            logger.info("No web results for query: %s", query)
            return []

        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for item in raw.get("results", []):
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").strip()
            if not url or not content or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                WebSearchResult(
                    title=(item.get("title") or url).strip(),
                    url=url,
                    content=content,
                    score=item.get("score"),
                    published_date=item.get("published_date"),
                    query=query,
                )
            )
        return results

    def extract(self, urls: list[str], *, query: str | None = None) -> list[WebPageExtract]:
        """Fetch cleaned page text for a few URLs (truncated to save tokens)."""
        urls = [u for u in dict.fromkeys(urls) if u.startswith(("http://", "https://"))]
        if not urls:
            return []
        payload: dict[str, Any] = {"urls": urls[:MAX_EXTRACT_URLS]}
        if query:
            payload["query"] = query
        try:
            raw = self._invoke(self.extract_tool, payload, "Tavily extract")
        except ToolException:
            return []
        extracts = []
        for item in raw.get("results", []):
            content = (item.get("raw_content") or "").strip()
            if item.get("url") and content:
                extracts.append(WebPageExtract(url=item["url"], content=content[:MAX_EXTRACT_CHARS]))
        return extracts

    def _invoke(self, tool: BaseTool, payload: dict[str, Any], description: str) -> dict[str, Any]:
        try:
            return call_with_retry(
                lambda: _raise_if_error(tool.invoke(payload)),
                should_retry=_is_retryable,
                description=description,
                **self._retry_kwargs,
            )
        except _TavilyCallError as exc:
            logger.warning("%s failed: status=%s", description, exc.status)
            raise _to_service_error(exc) from None
