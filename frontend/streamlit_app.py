"""ResearchPilot Streamlit app.

    python run.py frontend        (or: streamlit run frontend/streamlit_app.py)

The backend must be running (python run.py backend); BACKEND_URL selects it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from frontend.components import PAGES_KEY, system_state  # noqa: E402
from frontend.ui.components import system_status_card  # noqa: E402
from frontend.ui.theme import inject_theme  # noqa: E402
from frontend.views import history, knowledge_base, new_research, settings_page  # noqa: E402

# The logo image is scaled to the sidebar header; give the two-line wordmark room to stay readable.
LOGO_CSS = '<style>[data-testid="stSidebarHeader"] img { height: 3.1rem !important; max-width: 100%; }</style>'


def main() -> None:
    st.set_page_config(page_title="ResearchPilot", page_icon=str(ASSETS / "logo_icon.svg"), layout="wide")
    inject_theme()
    st.html(LOGO_CSS)
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "logo_icon.svg"))
    pages = {
        "research": st.Page(new_research.render, title="New Research", icon=":material/add_circle:", default=True),
        "history": st.Page(history.render, title="Research History", icon=":material/history:", url_path="history"),
        "knowledge": st.Page(knowledge_base.render, title="Knowledge Base", icon=":material/inventory_2:",
                             url_path="knowledge-base"),
        "settings": st.Page(settings_page.render, title="Settings", icon=":material/settings:", url_path="settings"),
    }
    st.session_state[PAGES_KEY] = pages
    navigation = st.navigation(list(pages.values()))
    with st.sidebar:
        system_status_card(system_state())
    navigation.run()


main()
