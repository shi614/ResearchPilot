"""PDF export of a finished research report (ReportLab).

The PDF is built from the stored report JSON (no LLM calls). It contains every
report section, inline citations linked to a numbered reference list, a
clearly marked verification status for each source, and the quality notes.
"""

from __future__ import annotations

import html
import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from app.models.domain import Citation, SourceType, VerificationStatus
from app.services.report_render import StoredReport

# Built-in PDF fonts only cover Windows-1252; map common symbols, then degrade safely.
_REPLACEMENTS = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~", "×": "x", "−": "-", "✓": "v"}
_CITATION = re.compile(r"\[((?:[WK]\d+)(?:\s*[,;]\s*[WK]\d+)*)\]")
_ID = re.compile(r"[WK]\d+")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])")
_CODE = re.compile(r"`([^`]+)`")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)")
_HEADING = re.compile(r"^#{1,6}\s+(.*)")

_STATUS_TEXT = {
    VerificationStatus.CORROBORATED: ("Corroborated by multiple sources", "#1b7f3b"),
    VerificationStatus.SINGLE_SOURCE: ("Single source - not independently corroborated", "#a15c00"),
    VerificationStatus.CONFLICTING: ("Conflicting - other sources disagree", "#b3261e"),
    VerificationStatus.UNVERIFIED: ("UNVERIFIED - could not be verified", "#b3261e"),
}


def to_pdf_text(text: str) -> str:
    """Make text safe for the built-in fonts (Windows-1252)."""
    for symbol, replacement in _REPLACEMENTS.items():
        text = text.replace(symbol, replacement)
    out = []
    for char in text:
        try:
            char.encode("cp1252")
            out.append(char)
        except UnicodeEncodeError:
            ascii_form = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode()
            out.append(ascii_form or "?")
    return "".join(out)


def inline_markup(text: str, known_ids: set[str]) -> str:
    """Escape text for ReportLab's mini-HTML, then apply bold/italic/code and citation links."""
    markup = html.escape(to_pdf_text(text), quote=False)
    markup = _BOLD.sub(r"<b>\1</b>", markup)
    markup = _ITALIC.sub(r"<i>\1</i>", markup)
    markup = _CODE.sub(r'<font face="Courier">\1</font>', markup)

    def link(match: re.Match[str]) -> str:
        parts = []
        for source_id in _ID.findall(match.group(1)):
            if source_id in known_ids:
                parts.append(f'<a href="#ref-{source_id}" color="#1a56db">{source_id}</a>')
            else:
                parts.append(source_id)
        return "[" + ", ".join(parts) + "]"

    return _CITATION.sub(link, markup)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontSize=10.5, leading=15, spaceAfter=6)
    return {
        "title": ParagraphStyle("RPTitle", parent=base["Title"], fontSize=20, leading=25, spaceAfter=6),
        "subtitle": ParagraphStyle("RPSub", parent=body, alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
        "meta": ParagraphStyle("RPMeta", parent=body, fontSize=9, textColor=colors.HexColor("#555555")),
        "h1": ParagraphStyle("RPH1", parent=base["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6,
                             textColor=colors.HexColor("#1f3a5f")),
        "h2": ParagraphStyle("RPH2", parent=base["Heading3"], fontSize=11.5, spaceBefore=8, spaceAfter=4),
        "body": body,
        "ref": ParagraphStyle("RPRef", parent=body, fontSize=9, leading=12.5, leftIndent=14, firstLineIndent=-14),
    }


class _PdfWriter:
    def __init__(self, stored: StoredReport) -> None:
        self.stored = stored
        self.known_ids = {c.id for c in stored.citations}
        self.styles = _styles()
        self.story: list[Flowable] = []

    def para(self, text: str, style: str = "body") -> None:
        self.story.append(Paragraph(inline_markup(text, self.known_ids), self.styles[style]))

    def heading(self, text: str, level: str = "h1") -> None:
        self.story.append(Paragraph(html.escape(to_pdf_text(text)), self.styles[level]))

    def items(self, entries: list[str], numbered: bool = False) -> None:
        if not entries:
            return
        flowables = [ListItem(Paragraph(inline_markup(e, self.known_ids), self.styles["body"])) for e in entries]
        if numbered:
            listing = ListFlowable(flowables, bulletType="1", bulletFormat="%s.", leftIndent=18,
                                   bulletFontSize=self.styles["body"].fontSize)
        else:
            listing = ListFlowable(flowables, bulletType="bullet", start="•", leftIndent=16, bulletFontSize=9)
        self.story.append(listing)

    def markdown_block(self, text: str) -> None:
        """Render paragraphs, bullet/numbered lists and sub-headings from model-written Markdown."""
        for block in re.split(r"\n\s*\n", text.strip()):
            lines = [line for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if all(_BULLET.match(line) for line in lines):
                self.items([_BULLET.match(line).group(1) for line in lines])
            elif all(_NUMBERED.match(line) for line in lines):
                self.items([_NUMBERED.match(line).group(1) for line in lines], numbered=True)
            elif _HEADING.match(lines[0]):
                self.heading(_HEADING.match(lines[0]).group(1), "h2")
                if lines[1:]:
                    self.markdown_block("\n".join(lines[1:]))
            else:
                self.para(" ".join(line.strip() for line in lines))

    def reference(self, number: int, citation: Citation) -> None:
        label, color = _STATUS_TEXT[citation.verification_status]
        kind = "Web source" if citation.source_type is SourceType.WEB else "Knowledge base"
        reference = html.escape(to_pdf_text(citation.reference), quote=False)
        if citation.reference.startswith(("http://", "https://")):
            reference = f'<a href="{html.escape(citation.reference)}" color="#1a56db">{reference}</a>'
        title = html.escape(to_pdf_text(citation.title), quote=False)
        location = kind if citation.reference == citation.title else f"{kind}: {reference}"
        self.story.append(
            Paragraph(
                f'<a name="ref-{citation.id}"/>{number}. <b>[{citation.id}]</b> {title}. {location}<br/>'
                f'<font color="{color}"><b>Verification:</b> {label}</font>',
                self.styles["ref"],
            )
        )

    def build(self, query: str, generated_at: datetime) -> list[Flowable]:
        report = self.stored.report
        self.story.append(Paragraph(html.escape(to_pdf_text(report.title)), self.styles["title"]))
        self.story.append(Paragraph("ResearchPilot - Multi-Agent AI Research Report", self.styles["subtitle"]))
        self.story.append(Spacer(1, 8))
        self.para(f"**Research question:** {query}", "meta")
        self.para(f"**Generated:** {generated_at:%d %B %Y, %H:%M} UTC", "meta")
        self.story.append(Spacer(1, 6))

        self.heading("Executive Summary")
        self.markdown_block(report.executive_summary)
        self.heading("Introduction")
        self.markdown_block(report.introduction)
        self.heading("Research Questions")
        self.items(report.research_questions, numbered=True)
        self.heading("Methodology")
        self.markdown_block(report.methodology)
        self.heading("Key Findings")
        self.items(report.key_findings)
        self.heading("Detailed Analysis")
        for section in report.detailed_analysis:
            self.heading(section.heading, "h2")
            self.markdown_block(section.content)
        self.heading("Limitations")
        self.markdown_block(report.limitations)
        self.heading("Conclusion")
        self.markdown_block(report.conclusion)

        self.heading("References")
        if self.stored.citations:
            for number, citation in enumerate(self.stored.citations, start=1):
                self.reference(number, citation)
        else:
            self.para("No sources were cited.")
        if self.stored.quality_notes:
            self.heading("Quality Notes")
            self.items(self.stored.quality_notes)
        return self.story


def _footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "Generated by ResearchPilot - verify important claims against the cited sources")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report_pdf(stored: StoredReport, path: Path, *, query: str, generated_at: datetime) -> Path:
    """Write the report PDF to `path` (atomically) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")  # unique: concurrent builds
    document = SimpleDocTemplate(
        str(temporary),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=to_pdf_text(stored.report.title),
        author="ResearchPilot",
        subject=to_pdf_text(query),
    )
    document.build(_PdfWriter(stored).build(query, generated_at), onFirstPage=_footer, onLaterPages=_footer)
    temporary.replace(path)
    return path
