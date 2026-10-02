"""Test doubles for the agent layer.

`ScriptedLLM` replaces Gemini at the gateway boundary. Responses are queued per
output schema and may be callables that build answers from the actual prompt
(e.g. citing the source IDs the agent really provided), so agent logic,
routing and state handling are exercised for real.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from app.config import Settings
from app.models.agent_outputs import (
    AnalysisNotes,
    ClaimCheck,
    Critique,
    CritiqueIssue,
    Finding,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    SourceAssessment,
    VerificationResult,
)
from app.models.domain import RetrievedChunk
from app.tools.web_search import WebSearchTools

Responder = Any  # a BaseModel instance, an Exception, or Callable[[str], BaseModel]
SOURCE_ID = re.compile(r"\[([WK]\d+)\]")


class ScriptedLLM:
    def __init__(
        self,
        structured: dict[type, list[Responder]] | None = None,
        tool_turns: list[AIMessage | Exception] | None = None,
    ) -> None:
        self.structured_responses = {k: list(v) for k, v in (structured or {}).items()}
        self.tool_turns = list(tool_turns or [])
        self._calls = 0
        self.operations: list[str] = []
        self.prompts: dict[str, list[str]] = {}

    @property
    def call_count(self) -> int:
        return self._calls

    def calls_for(self, operation_prefix: str) -> int:
        return sum(op.startswith(operation_prefix) for op in self.operations)

    def structured(self, schema: type, system: str, user: str, *, operation: str) -> Any:
        self._calls += 1
        self.operations.append(operation)
        self.prompts.setdefault(operation, []).append(user)
        queue = self.structured_responses.get(schema) or []
        if not queue:
            raise AssertionError(f"No scripted response left for {schema.__name__} ({operation})")
        response = queue.pop(0) if len(queue) > 1 else queue[0]  # last response repeats
        if isinstance(response, Exception):
            raise response
        return response(user) if callable(response) else response.model_copy(deep=True)

    def invoke_with_tools(self, messages: list[BaseMessage], tools: list, *, operation: str) -> AIMessage:
        self._calls += 1
        self.operations.append(operation)
        if not self.tool_turns:
            return AIMessage(content="Enough information gathered.")
        turn = self.tool_turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def tool_call_turn(*queries: str, extract: list[str] | None = None) -> AIMessage:
    calls = [
        {"name": "tavily_search", "args": {"query": q}, "id": f"call-{i}", "type": "tool_call"}
        for i, q in enumerate(queries)
    ]
    if extract:
        calls.append({"name": "tavily_extract", "args": {"urls": extract}, "id": "call-x", "type": "tool_call"})
    return AIMessage(content="", tool_calls=calls)


# --------------------------------------------------------------------------- Tavily stub


class FakeSearchTool:
    """Tavily search stand-in: 2 deterministic results per query (or `empty`)."""

    name = "tavily_search"

    def __init__(self, empty: bool = False, fail_with: dict | None = None) -> None:
        self.empty = empty
        self.fail_with = fail_with
        self.queries: list[str] = []

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        self.queries.append(query)
        if self.fail_with is not None:
            return self.fail_with
        if self.empty:
            return {"results": []}
        slug = re.sub(r"\W+", "-", query.lower()).strip("-")
        return {
            "results": [
                {"title": f"{query} — overview", "url": f"https://example.org/{slug}", "content": f"Facts about {query}.", "score": 0.9},
                {"title": f"{query} — study", "url": f"https://journal.example/{slug}", "content": f"Study on {query}.", "score": 0.8},
            ]
        }


class FakeExtractTool:
    name = "tavily_extract"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload["urls"])
        return {"results": [{"url": u, "raw_content": f"Full text of {u}"} for u in payload["urls"]]}


def fake_web(search: FakeSearchTool | None = None, extract: FakeExtractTool | None = None) -> WebSearchTools:
    return WebSearchTools(search or FakeSearchTool(), extract or FakeExtractTool(), sleep=lambda _s: None)  # type: ignore[arg-type]


class FakeKnowledge:
    def __init__(self, chunks: list[RetrievedChunk] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.queries: list[str] = []

    def has_documents(self) -> bool:
        return bool(self.chunks)

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return list(self.chunks)


def kb_chunk(text: str = "Internal report: panel costs fell 40%.", page: int | None = 3) -> RetrievedChunk:
    return RetrievedChunk(document_id="doc1", filename="internal.pdf", chunk_index=0, page=page, content=text, relevance=0.8)


# --------------------------------------------------------------------------- scripted agent outputs


def plan(use_kb: bool = True, queries: list[str] | None = None) -> ResearchPlan:
    return ResearchPlan(
        objective="Understand solar energy cost trends",
        research_questions=["How have solar costs changed?", "What drives adoption?"],
        search_queries=queries if queries is not None else ["solar cost trends", "solar adoption drivers"],
        use_knowledge_base=use_kb,
        scope_notes="Global, last decade",
    )


def ids_in(prompt: str) -> list[str]:
    return list(dict.fromkeys(SOURCE_ID.findall(prompt)))


def verification(prompt: str) -> VerificationResult:
    ids = ids_in(prompt)
    return VerificationResult(
        assessments=[SourceAssessment(source_id=i, relevance="high", reliability="high", reason="ok") for i in ids],
        claims=[ClaimCheck(claim="Costs fell", source_ids=ids[:2], status="corroborated")],
        conflicts=[],
        summary="Sources broadly agree.",
    )


def analysis(prompt: str) -> AnalysisNotes:
    ids = ids_in(prompt)
    return AnalysisNotes(
        answer_summary="Solar costs fell sharply.",
        key_findings=[Finding(statement="Costs fell", source_ids=ids[:2], confidence="high"),
                      Finding(statement="Bogus", source_ids=["W999"], confidence="low")],
        patterns=["Falling costs"],
        comparisons=["Sources agree"],
        gaps=["Regional data"],
    )


def report(prompt: str, extra_citation: str = "") -> ResearchReport:
    ids = ids_in(prompt.split("Evidence")[-1])
    first = f"[{ids[0]}]" if ids else ""
    second = f"[{ids[1]}]" if len(ids) > 1 else ""
    return ResearchReport(
        title="Solar Energy Cost Trends",
        executive_summary=f"Costs fell {first}.",
        introduction="Intro.",
        research_questions=["How have solar costs changed?"],
        methodology="Multi-agent research.",
        key_findings=[f"Costs fell {first}{second}{extra_citation}"],
        detailed_analysis=[ReportSection(heading="Costs", content=f"Details {first}.")],
        limitations="Limited data.",
        conclusion="Solar is cheaper.",
    )


def clean_critique() -> Critique:
    return Critique(issues=[], overall_assessment="Sound.")


def critique_with(severity: str = "high") -> Critique:
    return Critique(
        issues=[CritiqueIssue(severity=severity, category="unsupported_claim", location="Conclusion",
                              description="Claim not supported", suggestion="Soften it")],
        overall_assessment="Needs work.",
    )


def happy_llm(**overrides: list[Responder]) -> ScriptedLLM:
    structured: dict[type, list[Responder]] = {
        ResearchPlan: [plan()],
        VerificationResult: [verification],
        AnalysisNotes: [analysis],
        ResearchReport: [report],
        Critique: [clean_critique()],
    }
    for schema in list(structured):
        if schema.__name__ in overrides:
            structured[schema] = overrides[schema.__name__]
    return ScriptedLLM(structured, tool_turns=[tool_call_turn("solar cost trends", "solar adoption drivers")])


def agent_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


ProgressFn = Callable[[str, str, dict], None]
