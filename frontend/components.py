"""App-level helpers shared by the pages: API client access, error handling,
navigation, formatting and the user-facing system state.

Visual building blocks live in `frontend.ui` (theme + components).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, TypeVar

import streamlit as st

from frontend.api_client import ApiError, BackendUnavailable, ResearchPilotClient

T = TypeVar("T")

CLIENT_KEY = "api_client"
ACTIVE_SESSION_KEY = "active_session_id"
PAGES_KEY = "_pages"

SystemState = Literal["ready", "limited", "offline"]

# User preferences (Settings page), kept for the browser session.
PREF_DEFAULT_INSTRUCTIONS = "pref_default_instructions"
PREF_SHOW_ACTIVITY = "pref_show_activity"
PREF_COMPACT = "pref_compact"


def client() -> ResearchPilotClient:
    if CLIENT_KEY not in st.session_state:
        st.session_state[CLIENT_KEY] = ResearchPilotClient()
    return st.session_state[CLIENT_KEY]


def call(action: Callable[[], T], *, failure: str = "Something went wrong") -> T | None:
    """Run an API call; show a clear message instead of a stack trace on failure."""
    try:
        return action()
    except BackendUnavailable as exc:
        st.error(exc.message, icon=":material/cloud_off:")
    except ApiError as exc:
        st.error(f"{failure}: {exc.message}", icon=":material/error:")
    return None


def quietly(action: Callable[[], T]) -> T | None:
    """For secondary information: return None instead of showing an error."""
    try:
        return action()
    except ApiError:
        return None


def system_state(config: dict | None = None) -> SystemState:
    """User-facing status. Pass an already-fetched config to avoid a second request."""
    config = config if config is not None else quietly(lambda: client().config())
    if config is None:
        return "offline"
    if not config["research_enabled"] or not config["web_research_enabled"]:
        return "limited"
    return "ready"


def open_session(session_id: str | None) -> None:
    st.session_state[ACTIVE_SESSION_KEY] = session_id
    for key in [k for k in st.session_state if str(k).startswith(("report:", "pdf:", "modify_open", "view:"))]:
        del st.session_state[key]


def go_to(page: str) -> None:
    pages = st.session_state.get(PAGES_KEY)
    if pages and page in pages:
        st.switch_page(pages[page])
    else:
        st.rerun()


def local_time(value: str, fmt: str = "%d %b %Y, %H:%M") -> str:
    """Backend timestamps are UTC (naive ones included); show them in the viewer's local time."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone().strftime(fmt)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"
