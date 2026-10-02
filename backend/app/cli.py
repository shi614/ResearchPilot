"""Terminal front-end for the research workflow (pre-UI; also handy for demos).

    python run.py research "your question" [--instructions "..."] [--auto-approve]
    python run.py research --retry <run_id>
"""

from __future__ import annotations

import argparse
import uuid
from typing import Any

from app.agents.formatting import report_to_markdown
from app.config import get_settings
from app.database import Database
from app.graph.runner import RunOutcome
from app.models.agent_outputs import HumanDecision
from app.rag.embeddings import create_embeddings
from app.rag.vectorstore import KnowledgeBaseStore
from app.services.knowledge_service import KnowledgeService
from app.services.research_factory import create_research_engine


def _print_update(_thread_id: str, node: str, update: dict[str, Any]) -> None:
    for event in update.get("progress", []):
        print(f"  ✓ {event.node:<20} {event.message}")
    for err in update.get("errors", []):
        print(f"  ! {err.node:<20} {err.message}")


def _ask_decision(request: dict[str, Any]) -> HumanDecision:
    print("\n── Research complete — approval needed ─────────────────────────")
    sources = request["sources"]
    print(f"Sources: {sources['total']} ({sources['web']} web, {sources['knowledge_base']} knowledge base)")
    print("Research questions:\n" + "\n".join(f"  - {q}" for q in request["research_questions"]))
    if request.get("answer_summary"):
        print(f"Summary: {request['answer_summary']}")
    print("Key findings:\n" + "\n".join(f"  - {f}" for f in request["key_findings"]))
    options = "[a]pprove, [m]odify, [c]ancel" if request["can_modify"] else "[a]pprove, [c]ancel"
    choice = input(f"\n{options}: ").strip().lower()[:1]
    if choice == "m" and request["can_modify"]:
        return HumanDecision(action="modify", feedback=input("What should change? "))
    return HumanDecision(action="cancel" if choice == "c" else "approve")


def _print_result(outcome: RunOutcome, llm_calls: int) -> None:
    values = outcome.values
    print(f"\nStatus: {outcome.status}" + (f" — {outcome.error}" if outcome.error else ""))
    print(f"Gemini calls: {llm_calls} | web searches: {values.get('web_searches', 0)} | "
          f"revisions: {values.get('revision_count', 0)}")
    if outcome.retryable:
        print(f"Progress is saved. Resume later with: python run.py research --retry {outcome.thread_id}")
    report = values.get("final_report")
    if report is not None:
        print("\n" + report_to_markdown(report) + "\n\n## References")
        for citation in values.get("citations", []):
            print(f"[{citation.id}] {citation.title} — {citation.reference} ({citation.verification_status.value})")
        for note in values.get("quality_notes", []):
            print(f"Note: {note}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="run.py research")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--instructions")
    parser.add_argument("--auto-approve", action="store_true", help="Skip the approval prompt")
    parser.add_argument("--retry", metavar="RUN_ID", help="Resume a run stopped by an error/quota")
    args = parser.parse_args(argv)
    if not args.query and not args.retry:
        parser.error("a query or --retry RUN_ID is required")

    settings = get_settings()
    settings.ensure_directories()
    database = Database(settings.database_url)
    database.create_tables()
    embeddings = create_embeddings(settings) if settings.gemini_api_key else None
    knowledge = KnowledgeService(settings, database, KnowledgeBaseStore(settings.chroma_dir, embeddings))
    engine = create_research_engine(settings, knowledge, on_update=_print_update)

    if args.retry:
        thread_id = args.retry
        outcome = engine.runner.retry(thread_id)
    else:
        thread_id = uuid.uuid4().hex
        print(f"Run {thread_id} — model {settings.gemini_model}")
        outcome = engine.runner.start(thread_id, args.query, args.instructions)
    while outcome.status == "awaiting_approval":
        decision = HumanDecision(action="approve") if args.auto_approve else _ask_decision(outcome.approval_request or {})
        outcome = engine.runner.resume(thread_id, decision)
    _print_result(outcome, engine.llm.call_count)
