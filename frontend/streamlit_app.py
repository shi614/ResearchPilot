"""ResearchPilot Streamlit app.

    python run.py frontend        (or: streamlit run frontend/streamlit_app.py)

The backend must be running (python run.py backend); BACKEND_URL selects it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from frontend.api_client import ApiError  # noqa: E402
from frontend.components import PAGES_KEY, client  # noqa: E402
from frontend.views import history, knowledge_base, new_research, settings_page  # noqa: E402

STYLES = """
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  [data-testid="stSidebarHeader"] { padding-bottom: 0; }
</style>
"""


def _sidebar_status() -> None:
    with st.sidebar:
        st.markdown("### :material/travel_explore: ResearchPilot")
        st.caption("Multi-agent research assistant")
        try:
            health = client().health()
        except ApiError as exc:
            st.error("Backend offline", icon=":material/cloud_off:")
            st.caption(exc.message)
            return
        if health["status"] == "ok":
            st.success(f"Backend online · {health['gemini_model']}", icon=":material/cloud_done:")
        else:
            st.warning(f"Backend degraded · missing {', '.join(health['missing_keys'])}",
                       icon=":material/warning:")


def main() -> None:
    st.set_page_config(page_title="ResearchPilot", page_icon=":material/travel_explore:", layout="wide")
    st.html(STYLES)
    pages = {
        "research": st.Page(new_research.render, title="New Research", icon=":material/science:",
                            url_path="research", default=True),
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
