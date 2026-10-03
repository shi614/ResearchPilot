"""ResearchPilot design tokens and stylesheet.

Base colours/fonts live in `.streamlit/config.toml`; this module adds what the
Streamlit theme cannot express (cards, timeline, badges, keyed widget styling).
CSS is organised by section and built from the tokens below, so a colour or
radius change is made in one place.

Keyed containers (`st.container(key="composer")`) render with a `st-key-<key>`
class; styles target those classes instead of Streamlit internals where possible.
"""

from __future__ import annotations

import streamlit as st

TOKENS = {
    "navy": "#0f1b3d",
    "text": "#1f2a44",
    "muted": "#5f6b85",
    "subtle": "#8a94ab",
    "border": "#e4e7f0",
    "surface": "#ffffff",
    "canvas": "#f6f7fb",
    "accent": "#4f46e5",
    "accent_strong": "#4338ca",
    "accent_soft": "#eef0ff",
    "accent_border": "#c7cbfb",
    "success": "#067647",
    "success_soft": "#ecfdf3",
    "warning": "#b54708",
    "warning_soft": "#fffaeb",
    "danger": "#b42318",
    "danger_soft": "#fef3f2",
    "neutral_soft": "#f2f4f7",
    "radius": "14px",
    "radius_sm": "10px",
    "shadow": "0 1px 2px rgba(16,24,40,.04), 0 4px 14px rgba(16,24,40,.05)",
    "shadow_lg": "0 2px 4px rgba(16,24,40,.04), 0 12px 32px rgba(79,70,229,.10)",
}

_LAYOUT = """
.block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem; }
@media (max-width: 640px) { .block-container { padding-top: 4.5rem; } .rp-hero-title { font-size: 1.7rem; } }
h1, h2, h3 { color: {navy}; letter-spacing: -0.01em; }
.rp-ms { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; font-size: 1.15em;
         line-height: 1; display: inline-block; vertical-align: -0.18em; letter-spacing: normal;
         text-transform: none; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased;
         font-feature-settings: "liga"; }
"""

_TYPOGRAPHY = """
.rp-hero-title { font-size: 2rem; font-weight: 750; color: {navy}; margin: 0; letter-spacing: -0.02em; }
.rp-hero-subtitle { font-size: 1.12rem; font-weight: 600; color: {accent}; margin: .15rem 0 .35rem; }
.rp-hero-text { color: {muted}; font-size: .98rem; margin: 0 0 1.1rem; }
.rp-page-title { font-size: 1.75rem; font-weight: 750; color: {navy}; margin: 0; letter-spacing: -0.02em; }
.rp-session-title { font-size: 1.5rem; font-weight: 750; color: {navy}; line-height: 1.3; margin: 0;
                     letter-spacing: -0.01em; overflow-wrap: anywhere; }
.rp-page-subtitle { color: {muted}; font-size: 1rem; margin: .25rem 0 0; }
.rp-eyebrow { color: {accent}; font-size: .74rem; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; margin: 0 0 .25rem; }
.rp-section-title { font-size: 1.05rem; font-weight: 700; color: {navy}; margin: 0; }
.rp-section-sub { color: {muted}; font-size: .9rem; margin: .15rem 0 0; }
.rp-muted { color: {muted}; }
.rp-small { font-size: .82rem; color: {subtle}; }
"""

_CARDS = """
.rp-card { background: {surface}; border: 1px solid {border}; border-radius: {radius};
           box-shadow: {shadow}; padding: 1.15rem 1.25rem; }
.rp-card-accent { background: linear-gradient(180deg, {accent_soft} 0%, {surface} 85%);
                  border: 1px solid {accent_border}; box-shadow: {shadow_lg}; }
.rp-card-danger { background: {danger_soft}; border: 1px solid #fecdca; }
.rp-card h4 { margin: 0 0 .35rem; color: {navy}; font-size: 1.02rem; }
.rp-metric { background: {surface}; border: 1px solid {border}; border-radius: {radius_sm};
             padding: .8rem 1rem; box-shadow: {shadow}; height: 100%; }
.rp-metric-label { color: {muted}; font-size: .8rem; font-weight: 600; display: flex; gap: .35rem;
                   align-items: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rp-metric-value { color: {navy}; font-size: 1.5rem; font-weight: 750; line-height: 1.2; margin-top: .2rem; }
.rp-metric-hint { color: {subtle}; font-size: .78rem; }
"""

_BADGES = """
.rp-badge { display: inline-flex; align-items: center; gap: .4rem; border-radius: 999px;
            padding: .22rem .7rem; font-size: .8rem; font-weight: 600; white-space: nowrap;
            border: 1px solid transparent; }
.rp-badge::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }
.rp-badge-success { color: {success}; background: {success_soft}; border-color: #abefc6; }
.rp-badge-info { color: {accent}; background: {accent_soft}; border-color: {accent_border}; }
.rp-badge-warning { color: {warning}; background: {warning_soft}; border-color: #fedf89; }
.rp-badge-danger { color: {danger}; background: {danger_soft}; border-color: #fecdca; }
.rp-badge-neutral { color: {muted}; background: {neutral_soft}; border-color: {border}; }
"""

_STEPS = """
.rp-steps { display: flex; align-items: stretch; gap: .5rem; flex-wrap: wrap; }
.rp-step { flex: 1 1 150px; background: {surface}; border: 1px solid {border}; border-radius: {radius_sm};
           padding: .85rem .95rem; box-shadow: {shadow}; }
.rp-step-top { display: flex; align-items: center; gap: .5rem; margin-bottom: .3rem; }
.rp-step-icon { width: 2rem; height: 2rem; border-radius: 9px; background: {accent_soft}; color: {accent};
                display: inline-flex; align-items: center; justify-content: center; font-size: 1.05rem; }
.rp-step-num { color: {subtle}; font-size: .74rem; font-weight: 700; letter-spacing: .06em; }
.rp-step-label { color: {navy}; font-weight: 700; font-size: .95rem; }
.rp-step-text { color: {muted}; font-size: .8rem; line-height: 1.35; margin: 0; }
.rp-step-arrow { align-self: center; color: {accent_border}; font-size: 1.2rem; font-weight: 700; }
@media (max-width: 900px) { .rp-step-arrow { display: none; } }
"""

_TIMELINE = """
.rp-timeline { list-style: none; margin: 0; padding: 0; }
.rp-tl-item { display: flex; gap: .8rem; position: relative; padding-bottom: .9rem; }
.rp-tl-item:not(:last-child)::after { content: ""; position: absolute; left: .9rem; top: 1.95rem; bottom: .1rem;
                                     width: 2px; background: {border}; }
.rp-tl-item.done:not(:last-child)::after, .rp-tl-item.warning:not(:last-child)::after { background: {accent_border}; }
.rp-tl-dot { flex: 0 0 1.85rem; height: 1.85rem; border-radius: 50%; display: inline-flex; align-items: center;
             justify-content: center; font-size: .85rem; font-weight: 700; border: 2px solid {border};
             background: {surface}; color: {subtle}; z-index: 1; }
.rp-tl-item.done .rp-tl-dot { background: {accent}; border-color: {accent}; color: #fff; }
.rp-tl-item.warning .rp-tl-dot { background: {warning_soft}; border-color: #fdb022; color: {warning}; }
.rp-tl-item.active .rp-tl-dot, .rp-tl-item.waiting .rp-tl-dot { border-color: {accent}; color: {accent};
                                                                 background: {accent_soft}; }
.rp-tl-item.failed .rp-tl-dot { background: {danger_soft}; border-color: {danger}; color: {danger}; }
.rp-tl-item.skipped .rp-tl-dot { border-style: dashed; }
.rp-tl-body { padding-top: .2rem; min-width: 0; }
.rp-tl-label { color: {navy}; font-weight: 650; font-size: .93rem; line-height: 1.2; }
.rp-tl-item.pending .rp-tl-label, .rp-tl-item.skipped .rp-tl-label { color: {subtle}; font-weight: 550; }
.rp-tl-note { color: {muted}; font-size: .78rem; margin-top: .1rem; }
.rp-tl-item.active .rp-tl-note, .rp-tl-item.waiting .rp-tl-note { color: {accent}; font-weight: 600; }
.rp-tl-item.failed .rp-tl-note { color: {danger}; font-weight: 600; }
.rp-spinner { width: .8rem; height: .8rem; border: 2px solid {accent_border}; border-top-color: {accent};
              border-radius: 50%; display: inline-block; animation: rp-spin 1s linear infinite; }
@keyframes rp-spin { to { transform: rotate(360deg); } }
"""

_RESEARCH_CARDS = """
.rp-rcard-title { color: {navy}; font-weight: 700; font-size: 1.02rem; line-height: 1.3; margin: .5rem 0 .25rem;
                  overflow-wrap: anywhere; }
.rp-rcard-desc { color: {muted}; font-size: .86rem; margin: 0 0 .7rem; overflow-wrap: anywhere;
                 display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.rp-rcard-meta { display: flex; flex-wrap: wrap; gap: .9rem; color: {subtle}; font-size: .8rem; }
.rp-doc-row { display: flex; align-items: center; gap: .85rem; min-width: 0; }
.rp-doc-icon { flex: 0 0 2.4rem; height: 2.4rem; border-radius: 10px; display: inline-flex; align-items: center;
               justify-content: center; font-size: .72rem; font-weight: 800; color: {accent}; background: {accent_soft}; }
.rp-doc-name { color: {navy}; font-weight: 650; overflow-wrap: anywhere; }
.rp-doc-meta { color: {subtle}; font-size: .8rem; }
.rp-source { border-left: 3px solid {accent_border}; padding: .15rem 0 .15rem .8rem; margin-bottom: .75rem; }
.rp-source-title { color: {navy}; font-weight: 600; font-size: .92rem; overflow-wrap: anywhere; }
.rp-source-ref { color: {muted}; font-size: .8rem; overflow-wrap: anywhere; }
.rp-source-ref a { color: {accent}; text-decoration: none; }
"""

_EMPTY = """
.rp-empty { text-align: center; background: {surface}; border: 1px dashed {accent_border}; border-radius: {radius};
            padding: 2.4rem 1.5rem 1.2rem; }
.rp-empty-icon { width: 3rem; height: 3rem; border-radius: 14px; background: {accent_soft}; color: {accent};
                 display: inline-flex; align-items: center; justify-content: center; font-size: 1.4rem; }
.rp-empty h3 { margin: .8rem 0 .3rem; font-size: 1.15rem; color: {navy}; }
.rp-empty p { color: {muted}; margin: 0; }
"""

_WIDGETS = """
/* Keyed containers styled as cards (st.container(key=...)) */
.st-key-composer, [class*="st-key-card"] { background: {surface}; border: 1px solid {border};
    border-radius: {radius}; box-shadow: {shadow}; padding: 1.1rem 1.25rem 1.2rem; }
.st-key-composer { border-color: {accent_border}; box-shadow: {shadow_lg}; padding: 1.4rem 1.5rem 1.5rem; }
[class*="st-key-accentcard"] { background: linear-gradient(180deg, {accent_soft} 0%, {surface} 70%);
    border: 1px solid {accent_border}; border-radius: {radius}; box-shadow: {shadow_lg}; padding: 1.3rem 1.4rem; }
.st-key-composer textarea { font-size: 1.02rem; }
.stTextArea textarea, .stTextInput input { border-radius: {radius_sm}; }
/* Buttons: primary = accent, secondary = white with border, danger = keyed */
.stButton button, .stDownloadButton button, .stFormSubmitButton button, [data-testid="stPopover"] button {
    border-radius: {radius_sm}; font-weight: 600; min-height: 2.6rem; }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"],
.stFormSubmitButton button[kind="primary"] { box-shadow: 0 1px 2px rgba(79,70,229,.25); }
[class*="st-key-danger"] button { color: {danger}; border-color: #fecdca; background: {surface}; }
[class*="st-key-danger"] button:hover { background: {danger_soft}; border-color: {danger}; color: {danger}; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed {accent_border}; background: {accent_soft};
    border-radius: {radius_sm}; }
[data-testid="stExpander"] details { border-radius: {radius_sm}; }
"""

_SIDEBAR = """
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] { margin-top: auto; padding-bottom: 1.25rem; }
[data-testid="stSidebarNav"] a span { font-weight: 550; }
.rp-sys { background: {surface}; border: 1px solid {border}; border-radius: {radius_sm}; padding: .75rem .85rem;
          box-shadow: {shadow}; }
.rp-sys-title { display: flex; align-items: center; gap: .45rem; color: {navy}; font-weight: 700; font-size: .88rem; }
.rp-sys-dot { width: .55rem; height: .55rem; border-radius: 50%; background: {success};
              box-shadow: 0 0 0 3px {success_soft}; }
.rp-sys.warn .rp-sys-dot { background: #f79009; box-shadow: 0 0 0 3px {warning_soft}; }
.rp-sys.down .rp-sys-dot { background: {danger}; box-shadow: 0 0 0 3px {danger_soft}; }
.rp-sys-text { color: {muted}; font-size: .78rem; margin-top: .2rem; }
"""

_REPORT = """
.rp-report { max-width: 820px; }
.rp-report-text p, .rp-report-text li { line-height: 1.65; }
"""


def stylesheet() -> str:
    css = "".join((_LAYOUT, _TYPOGRAPHY, _CARDS, _BADGES, _STEPS, _TIMELINE, _RESEARCH_CARDS, _EMPTY,
                   _WIDGETS, _SIDEBAR, _REPORT))
    for name, value in TOKENS.items():
        css = css.replace("{" + name + "}", value)
    return f"<style>{css}</style>"


def inject_theme() -> None:
    st.html(stylesheet())
