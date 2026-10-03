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

from frontend.api_client import ApiError  # noqa: E402
from frontend.components import PAGES_KEY, client  # noqa: E402
from frontend.views import history, knowledge_base, new_research, settings_page  # noqa: E402

STYLES = """
<style>
  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }
  .rp-subtitle { color: #5b6475; font-size: 1.05rem; margin: -0.5rem 0 1.5rem 0; }
  .rp-section, .rp-eyebrow {
    color: #5b6475; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; margin: 1.1rem 0 0.35rem 0;
  }
  .rp-eyebrow { margin: 0 0 -0.4rem 0; }
  .rp-muted { color: #8a93a3; font-size: 0.8rem; margin-left: 0.35rem; }
  [data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
  [data-testid="stMetricLabel"] p { color: #5b6475; }
  [data-testid="stSidebarHeader"] { padding-bottom: 0.5rem; }
  [data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
</style>
"""


def _sidebar_status() -> None:
    with st.sidebar:
        try:
            health = client().health()
        except ApiError:
            st.badge("Offline", color="red", icon=":material/cloud_off:",
                     help="ResearchPilot's service isn't reachable. Start it with `python run.py backend`.")
            return
        if health["status"] == "ok":
            st.badge("Online", color="green", icon=":material/check_circle:")
        else:
            st.badge("Limited", color="orange", icon=":material/warning:",
                     help="Some features are unavailable. See Settings.")


def main() -> None:
    st.set_page_config(page_title="ResearchPilot", page_icon=":material/travel_explore:", layout="wide")
    st.html(STYLES)
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "logo_icon.svg"))
    pages = {
        "research": st.Page(new_research.render, title="New Research", icon=":material/science:", default=True),
        "history": st.Page(history.render, title="Research History", icon=":material/history:", url_path="history"),
        "knowledge": st.Page(knowledge_base.render, title="Knowledge Base", icon=":material/library_books:",
                             url_path="knowledge-base"),
        "settings": st.Page(settings_page.render, title="Settings", icon=":material/settings:", url_path="settings"),
    }
    st.session_state[PAGES_KEY] = pages
    navigation = st.navigation(list(pages.values()))
    _sidebar_status()
    navigation.run()


main()
