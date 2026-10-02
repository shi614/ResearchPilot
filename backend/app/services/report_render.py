"""Serialise a finished report (for storage and the API) and render it as Markdown.

The stored JSON is the single source for the API, the UI and the PDF export.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.agents.formatting import report_to_markdown
from app.models.agent_outputs import ResearchReport
from app.models.domain import Citation, VerificationStatus

_STATUS_LABEL = {
    VerificationStatus.CORROBORATED: "corroborated by multiple sources",
    VerificationStatus.SINGLE_SOURCE: "single source",
    VerificationStatus.CONFLICTING: "conflicting sources",
    VerificationStatus.UNVERIFIED: "unverified",
}


class StoredReport(BaseModel):
    report: ResearchReport
    citations: list[Citation]
    quality_notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_state(cls, values: dict[str, Any]) -> StoredReport:
        return cls(
            report=values["final_report"],
            citations=values.get("citations", []),
            quality_notes=values.get("quality_notes", []),
        )

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> StoredReport:
        return cls.model_validate(json.loads(data))


def render_markdown(stored: StoredReport) -> str:
    """Full report: body, numbered references with verification status, quality notes."""
    parts = [report_to_markdown(stored.report), "## References"]
    if stored.citations:
        for citation in stored.citations:
            kind = "Web" if citation.source_type.value == "web" else "Knowledge base"
            parts.append(
                f"- **[{citation.id}]** {citation.title}. {kind}: {citation.reference} "
                f"_({_STATUS_LABEL[citation.verification_status]})_"
            )
    else:
        parts.append("_No sources were cited._")
    if stored.quality_notes:
        parts.append("## Quality Notes")
        parts.extend(f"- {note}" for note in stored.quality_notes)
    return "\n\n".join(parts)
