"""Web research agent: LLM tool calling over Tavily search/extract.

The LLM decides which searches to run (round 1) and whether to refine or
extract pages after seeing the results (round 2). Tool execution goes through
`WebSearchTools`, which enforces retries, error mapping and normalisation.
A hard budget caps searches per iteration to protect Tavily credits.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, error, merge, progress
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.state import ResearchState
from app.models.domain import Source, SourceType, WebSearchResult

logger = logging.getLogger(__name__)

NODE = "web_research"
SEARCH_TOOL = "tavily_search"
EXTRACT_TOOL = "tavily_extract"
EXTRACT_CHARS_KEPT = 2500


class WebResearchAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps
        settings = deps.settings
        self._max_rounds = settings.max_research_tool_rounds
        self._max_searches = settings.max_web_searches

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        run = _WebResearchRun(self._deps, state, self._max_searches)
        counter = CallCounter(self._deps.llm)
        try:
            run.execute(self._max_rounds)
        except RateLimitError:
            raise  # Gemini quota: stop; the run resumes from the checkpoint
        except Exception as exc:  # noqa: BLE001 - an optional branch must not kill the run
            logger.exception("Unexpected web research error")
            run.updates.append(error(NODE, f"Web research error ({type(exc).__name__}); partial results kept"))
        return merge(
            {
                "web_sources": run.sources,
                "searched_queries": run.searched,
                "web_searches": run.search_count,
                "llm_calls": counter.used,
            },
            *run.updates,
            progress(
                NODE,
                f"{run.search_count} web searches ({run.mode}), {len(run.sources)} new sources"
                + (f", {run.extract_count} pages extracted" if run.extract_count else ""),
            ),
        )


class _WebResearchRun:
    """State for one execution of the web research node."""

    def __init__(self, deps: AgentDependencies, state: ResearchState, max_searches: int) -> None:
        assert deps.web is not None
        self.deps = deps
        self.web = deps.web
        self.state = state
        self.max_searches = max_searches
        self.plan = state["research_plan"]
        existing = state.get("web_sources", [])
        self.known_urls = {s.url for s in existing if s.url}
        self.next_id = len(existing) + 1
        self.already_searched = {q.lower() for q in state.get("searched_queries", [])}
        self.sources: list[Source] = []
        self.searched: list[str] = []
        self.search_count = 0
        self.extract_count = 0
        self.updates: list[dict[str, Any]] = []
        self.mode = "LLM tool calling"

    # ------------------------------------------------------------------ main loop

    def execute(self, max_rounds: int) -> None:
        messages: list[BaseMessage] = [
            SystemMessage(prompts.WEB_RESEARCHER),
            HumanMessage(self._task_prompt()),
        ]
        for round_number in range(1, max_rounds + 1):
            try:
                ai = self.deps.llm.invoke_with_tools(
                    messages, self.web.tools, operation=f"web research round {round_number}"
                )
            except RateLimitError:
                raise
            except ExternalServiceError as exc:
                self.updates.append(error(NODE, f"Tool-calling failed: {exc.message}"))
                break
            if not ai.tool_calls:
                break
            messages.append(ai)
            messages.extend(self._run_tool_calls(ai.tool_calls))
            if self.search_count >= self.max_searches:
                break

        if self.search_count == 0:
            # The LLM made no usable tool calls (or failed): run the planner's queries directly.
            self.mode = "planned queries"
            for query in self.plan.search_queries:
                self._search(query)

    def _task_prompt(self) -> str:
        questions = "\n".join(f"- {q}" for q in self.plan.research_questions)
        lines = [
            f"Objective: {self.plan.objective}",
            f"Research questions:\n{questions}",
            "Suggested queries: " + "; ".join(self.plan.search_queries),
            f"Budget: at most {self.max_searches} searches in total.",
        ]
        if self.already_searched:
            lines.append("Already searched (do not repeat): " + "; ".join(sorted(self.already_searched)))
        if self.state.get("human_feedback"):
            lines.append(f"User feedback to address: {self.state['human_feedback']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ tool execution

    def _run_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
        """Execute tool calls. Every call gets a ToolMessage reply, as the API requires."""
        replies = []
        for call in tool_calls:
            name, args = call.get("name"), call.get("args") or {}
            if name == SEARCH_TOOL:
                content = self._search(str(args.get("query", "")), time_range=args.get("time_range"))
            elif name == EXTRACT_TOOL:
                content = self._extract(list(args.get("urls") or []))
            else:
                content = f"Unknown tool '{name}'."
            replies.append(ToolMessage(content=content, tool_call_id=call.get("id") or name or "call"))
        return replies

    def _search(self, query: str, time_range: str | None = None) -> str:
        query = " ".join(query.split())
        if not query:
            return "Empty query ignored."
        if query.lower() in self.already_searched:
            return "Already searched; skipped."
        if self.search_count >= self.max_searches:
            return "Search budget exhausted; skipped."
        if time_range not in ("day", "week", "month", "year"):
            time_range = None

        self.already_searched.add(query.lower())
        self.searched.append(query)
        self.search_count += 1
        try:
            results = self.web.search(query, time_range=time_range)
        except (ExternalServiceError, ValueError) as exc:
            # Tavily failures (incl. credit/rate limits) degrade web research only.
            message = getattr(exc, "message", str(exc))
            self.updates.append(error(NODE, f"Search '{query}' failed: {message}"))
            return f"Search failed: {message}"

        new_sources = [self._add_source(result) for result in results]
        added = [s for s in new_sources if s is not None]
        if not results:
            return "No results."
        summary = [
            {"id": s.id, "title": s.title, "url": s.url, "snippet": s.content[:300]} for s in added
        ]
        return json.dumps(summary or "Only duplicates of earlier results.", ensure_ascii=False)

    def _add_source(self, result: WebSearchResult) -> Source | None:
        if result.url in self.known_urls:
            return None
        self.known_urls.add(result.url)
        source = Source(
            id=f"W{self.next_id}",
            source_type=SourceType.WEB,
            title=result.title,
            url=result.url,
            content=result.content,
            query=result.query,
            score=result.score,
        )
        self.next_id += 1
        self.sources.append(source)
        return source

    def _extract(self, urls: list[str]) -> str:
        by_url = {s.url: s for s in self.sources}
        urls = [u for u in urls if u in by_url][:2]  # only enrich sources found in this run
        if not urls:
            return "Extraction is only available for URLs returned by searches in this step."
        try:
            pages = self.web.extract(urls, query=self.plan.objective)
        except ExternalServiceError as exc:
            self.updates.append(error(NODE, f"Page extraction failed: {exc.message}"))
            return f"Extraction failed: {exc.message}"
        for page in pages:
            source = by_url[page.url]
            source.content = f"{source.content}\n\n[Extracted page text]\n{page.content[:EXTRACT_CHARS_KEPT]}"
            self.extract_count += 1
        return f"Extracted {len(pages)} page(s); content added to the corresponding sources."
