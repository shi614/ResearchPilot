"""HTTP client for the ResearchPilot backend.

The frontend never imports backend code: everything goes through the REST API,
so the UI shows exactly the state the backend reports. Failures become
`ApiError` / `BackendUnavailable` with messages safe to show to users.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_TIMEOUT_SECONDS = 600.0  # processing waits on the embedding rate limiter


def backend_url() -> str:
    """BACKEND_URL from the environment or .env (only this one value is read from .env)."""
    url = os.environ.get("BACKEND_URL") or dotenv_values(PROJECT_ROOT / ".env").get("BACKEND_URL")
    return (url or DEFAULT_BACKEND_URL).rstrip("/")


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BackendUnavailable(ApiError):
    """The backend could not be reached at all."""


class ResearchPilotClient:
    def __init__(self, base_url: str | None = None, *, http: httpx.Client | None = None,
                 timeout: float = 30.0) -> None:
        self.base_url = (base_url or backend_url()).rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------ transport

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise BackendUnavailable("The backend took too long to respond. Please try again.") from exc
        except httpx.TransportError as exc:
            raise BackendUnavailable(
                f"Cannot reach the ResearchPilot backend at {self.base_url}. "
                "Start it with `python run.py backend`."
            ) from exc
        if response.status_code >= 400:
            raise ApiError(self._error_message(response), response.status_code)
        return response

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return f"Backend error ({response.status_code})."
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, list):  # FastAPI validation errors
            return "; ".join(str(item.get("msg", item)) for item in detail) or "Invalid request."
        return str(detail or f"Backend error ({response.status_code}).")

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, path, **kwargs).json()

    # ------------------------------------------------------------------ health & settings

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def config(self) -> dict[str, Any]:
        return self._json("GET", "/health/config")

    def model_check(self, *, probe: bool = False) -> dict[str, Any]:
        params = {"probe": "true", "refresh": "true"} if probe else {}
        return self._json("GET", "/health/models", params=params, timeout=120.0)

    # ------------------------------------------------------------------ knowledge base

    def list_documents(self) -> list[dict[str, Any]]:
        return self._json("GET", "/knowledge/documents")

    def upload_document(self, filename: str, data: bytes) -> dict[str, Any]:
        return self._json("POST", "/knowledge/documents", files={"file": (filename, data)},
                          timeout=UPLOAD_TIMEOUT_SECONDS)

    def process_document(self, document_id: str) -> dict[str, Any]:
        return self._json("POST", f"/knowledge/documents/{document_id}/process", timeout=UPLOAD_TIMEOUT_SECONDS)

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/knowledge/documents/{document_id}")

    # ------------------------------------------------------------------ research

    def start_research(self, query: str, instructions: str | None = None) -> dict[str, Any]:
        return self._json("POST", "/research", json={"query": query, "instructions": instructions or None})

    def list_research(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._json("GET", "/research", params={"limit": limit})

    def get_research(self, session_id: str, after_event: int = 0) -> dict[str, Any]:
        return self._json("GET", f"/research/{session_id}", params={"after_event": after_event})

    def decide(self, session_id: str, action: str, feedback: str | None = None) -> dict[str, Any]:
        return self._json("POST", f"/research/{session_id}/decision", json={"action": action, "feedback": feedback})

    def retry(self, session_id: str) -> dict[str, Any]:
        return self._json("POST", f"/research/{session_id}/retry")

    def delete_research(self, session_id: str) -> None:
        self._request("DELETE", f"/research/{session_id}")

    def get_report(self, session_id: str) -> dict[str, Any]:
        return self._json("GET", f"/research/{session_id}/report")

    def get_report_pdf(self, session_id: str) -> bytes:
        return self._request("GET", f"/research/{session_id}/report.pdf", timeout=120.0).content
