"""Web-search tool tests. Tavily is stubbed at the tool boundary; normalisation,
retries and error mapping are exercised for real."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import ToolException

from app.config import Settings
from app.exceptions import ConfigurationError, ExternalServiceError, RateLimitError
from app.tools.web_search import MAX_QUERY_CHARS, WebSearchTools


class StubTool:
    """Replays queued responses; an Exception entry is raised, anything else returned."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.payloads.append(payload)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tavily_error(status: int) -> dict[str, Any]:
    """langchain-tavily returns API failures as {"error": ValueError("Error <status>: ...")}."""
    return {"error": ValueError(f"Error {status}: something went wrong")}


SEARCH_RESPONSE = {
    "query": "solar adoption",
    "results": [
        {"title": "IEA report", "url": "https://iea.org/a", "content": "Solar grew 30%.", "score": 0.91},
        {"title": "Dup", "url": "https://iea.org/a", "content": "Same URL again", "score": 0.5},
        {"title": "", "url": "https://news.example/b", "content": "Prices fell.", "score": 0.7},
        {"title": "No content", "url": "https://empty.example", "content": "", "score": 0.6},
    ],
}


def make_tools(search: StubTool, extract: StubTool | None = None) -> WebSearchTools:
    return WebSearchTools(search, extract or StubTool(), sleep=lambda _s: None)  # type: ignore[arg-type]


def test_search_normalises_and_deduplicates_results() -> None:
    tools = make_tools(StubTool(SEARCH_RESPONSE))
    results = tools.search("  solar   adoption ")
    assert [r.url for r in results] == ["https://iea.org/a", "https://news.example/b"]
    assert results[0].score == 0.91 and results[0].query == "solar adoption"
    assert results[1].title == "https://news.example/b"  # falls back to URL when untitled
    assert all(r.source_type.value == "web" for r in results)


def test_search_passes_optional_filters_and_truncates_long_queries() -> None:
    stub = StubTool({"results": []})
    make_tools(stub).search("x" * 1000, time_range="year", include_domains=["who.int"])
    payload = stub.payloads[0]
    assert len(payload["query"]) == MAX_QUERY_CHARS
    assert payload["time_range"] == "year" and payload["include_domains"] == ["who.int"]


def test_empty_results_return_empty_list() -> None:
    # langchain-tavily raises ToolException when a search finds nothing
    assert make_tools(StubTool(ToolException("No search results found"))).search("obscure") == []


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_tools(StubTool()).search("   ")


def test_rate_limit_is_retried_then_succeeds() -> None:
    stub = StubTool(tavily_error(429), SEARCH_RESPONSE)
    assert len(make_tools(stub).search("solar")) == 2
    assert len(stub.payloads) == 2


def test_persistent_rate_limit_raises_rate_limit_error() -> None:
    stub = StubTool(tavily_error(432), tavily_error(432), tavily_error(432))
    with pytest.raises(RateLimitError):
        make_tools(stub).search("solar")


def test_invalid_key_fails_fast_without_retry() -> None:
    stub = StubTool(tavily_error(401))
    with pytest.raises(ExternalServiceError, match="TAVILY_API_KEY"):
        make_tools(stub).search("solar")
    assert len(stub.payloads) == 1


def test_extract_filters_urls_and_truncates_content() -> None:
    extract = StubTool(
        {"results": [{"url": "https://a.example", "raw_content": "A" * 20000}], "failed_results": []}
    )
    tools = make_tools(StubTool(), extract)
    pages = tools.extract(["https://a.example", "https://a.example", "ftp://bad", "not-a-url"])
    assert extract.payloads[0]["urls"] == ["https://a.example"]
    assert len(pages) == 1 and len(pages[0].content) == 8000


def test_extract_with_no_valid_urls_makes_no_call() -> None:
    extract = StubTool()
    assert make_tools(StubTool(), extract).extract(["javascript:alert(1)"]) == []
    assert extract.payloads == []


def test_from_settings_requires_api_key() -> None:
    with pytest.raises(ConfigurationError):
        WebSearchTools.from_settings(Settings(_env_file=None))


def test_from_settings_exposes_langchain_tools_for_tool_calling() -> None:
    tools = WebSearchTools.from_settings(
        Settings(_env_file=None, tavily_api_key="tvly-test", tavily_max_results=3)
    )
    assert [tool.name for tool in tools.tools] == ["tavily_search", "tavily_extract"]
    assert tools.search_tool.max_results == 3  # type: ignore[attr-defined]
