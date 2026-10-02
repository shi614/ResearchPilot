"""Revision agent: fixes the critic's high/medium issues using only the existing evidence."""

from __future__ import annotations

from typing import Any

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, error, merge, progress
from app.agents.formatting import format_evidence, report_to_markdown
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.state import ResearchState
from app.models.agent_outputs import ResearchReport

NODE = "revision"


class RevisionAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        revision = state.get("revision_count", 0) + 1
        issues = [i for i in state["critique"].issues if i.severity in ("high", "medium")]
        issue_text = "\n".join(
            f"- [{i.severity}/{i.category}] {i.location}: {i.description} → {i.suggestion}" for i in issues
        )
        counter = CallCounter(self._deps.llm)
        user = (
            f"Issues to fix:\n{issue_text}\n\nCurrent report:\n{report_to_markdown(state['draft_report'])}\n\n"
            f"Evidence (cite only these IDs):\n{format_evidence(state.get('evidence', []), max_chars=500)}"
        )
        try:
            revised = self._deps.llm.structured(ResearchReport, prompts.REVISER, user, operation="revision")
        except RateLimitError:
            raise
        except ExternalServiceError as exc:
            # Keep the draft; the counter still advances so the loop cannot repeat forever.
            return merge(
                {"revision_count": revision, "llm_calls": counter.used},
                error(NODE, f"Revision failed; keeping the previous draft: {exc.message}"),
                progress(NODE, f"Revision {revision} failed"),
            )
        return merge(
            {"draft_report": revised, "revision_count": revision, "llm_calls": counter.used},
            progress(NODE, f"Revision {revision}: addressed {len(issues)} issue(s)"),
        )
