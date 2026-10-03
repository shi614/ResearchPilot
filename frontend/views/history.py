"""Research history: browse previous sessions, open them, delete them."""

from __future__ import annotations

import streamlit as st

from frontend.components import call, client, go_to, local_time, open_session, page_header, plural
from frontend.workflow import STATUS_BADGES


def render() -> None:
    page_header("Research History", "Revisit earlier research, continue paused runs or download reports.")
    sessions = call(lambda: client().list_research(), failure="Could not load research history")
    if sessions is None:
        return
    if not sessions:
        st.info("No research yet. Start your first one from **New Research**.", icon=":material/history:")
        return

    rows = [
        {
            "Date": local_time(s["created_at"]),
            "Topic": s["query"],
            "Report": s.get("title") or "—",
            "Status": STATUS_BADGES.get(s["status"], (s["status"],))[0],
        }
        for s in sessions
    ]
    selection = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key="history_table")
    selected = selection.selection.rows if selection else []
    if not selected:
        st.caption(f"{plural(len(sessions), 'research session')}. Select one to open or delete it.")
        return

    session = sessions[selected[0]]
    with st.container(border=True):
        st.markdown(f"**{session.get('title') or session['query']}**")
        open_col, delete_col, confirm_col = st.columns([2, 2, 3])
        if open_col.button("Open", type="primary", icon=":material/open_in_new:", width="stretch", key="open"):
            open_session(session["id"])
            go_to("research")
        confirm = confirm_col.checkbox("Confirm permanent deletion", key=f"confirm_{session['id']}")
        if delete_col.button("Delete", icon=":material/delete:", width="stretch", disabled=not confirm,
                             key="delete"):
            deleted = call(lambda: client().delete_research(session["id"]) or True, failure="Could not delete")
            if deleted:
                if st.session_state.get("active_session_id") == session["id"]:
                    open_session(None)
                st.toast("Research deleted", icon=":material/delete:")
                st.rerun()
