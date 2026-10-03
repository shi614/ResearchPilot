"""ResearchPilot design tokens and stylesheet (light, warm-ivory theme).

Base colours/fonts live in `.streamlit/config.toml`; this module adds what the
Streamlit theme cannot express (cards, timeline, badges, flow, keyed widget styling).
CSS is organised by section and built from the tokens below, so a colour or
radius change is made in one place.

Keyed containers (`st.container(key="composer")`) render with a `st-key-<key>`
class; styles target those classes instead of Streamlit internals where possible.
"""

from __future__ import annotations

import streamlit as st

TOKENS = {
    # palette
    "ivory": "#F8F7F3",       # app background
    "surface": "#FFFFFF",     # cards
    "navy": "#172033",        # primary text / headings
    "teal": "#167D8D",        # primary action
    "teal_dark": "#126877",   # primary hover
    "teal_soft": "#DDF2F3",   # light primary surface
    "gold": "#D6A84F",        # accent - use sparingly
    "slate": "#667085",       # secondary text
    "sage": "#4F8A70",        # success
    "red": "#C75C5C",         # error
    "border": "#E5E7EB",
    # derived tints (same hues, lighter)
    "teal_line": "#B9E1E4",
    "sage_soft": "#E6F0EB",
    "red_soft": "#F8E8E8",
    "gold_soft": "#F7EEDB",
    "gray_soft": "#F2F2EF",
    "subtle": "#98A2B3",
    # shape
    "radius": "12px",
    "radius_sm": "8px",
    "shadow": "0 1px 2px rgba(23,32,51,.04), 0 2px 8px rgba(23,32,51,.04)",
    "shadow_lg": "0 1px 3px rgba(23,32,51,.05), 0 8px 24px rgba(23,32,51,.06)",
}

_LAYOUT = """
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
@media (max-width: 640px) { .block-container { padding-top: 4.5rem; } .rp-hero-title { font-size: 1.8rem; } }
h1, h2, h3 { color: {navy}; letter-spacing: -0.01em; }
.rp-ms { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; font-size: 1.15em;
         line-height: 1; display: inline-block; vertical-align: -0.18em; letter-spacing: normal;
         text-transform: none; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased;
         font-feature-settings: "liga"; }
"""

_TYPOGRAPHY = """
.rp-hero-title { font-size: 2.35rem; font-weight: 750; color: {navy}; margin: 0; letter-spacing: -0.025em;
                 line-height: 1.15; }
.rp-hero-text { color: {slate}; font-size: 1.05rem; line-height: 1.55; margin: .55rem 0 1.25rem; max-width: 760px; }
.rp-page-title { font-size: 1.75rem; font-weight: 750; color: {navy}; margin: 0; letter-spacing: -0.02em; }
.rp-session-title { font-size: 1.45rem; font-weight: 700; color: {navy}; line-height: 1.3; margin: 0;
                    letter-spacing: -0.01em; overflow-wrap: anywhere; }
.rp-page-subtitle { color: {slate}; font-size: 1rem; line-height: 1.5; margin: .3rem 0 0; }
.rp-eyebrow { color: {teal}; font-size: .74rem; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; margin: 0 0 .3rem; }
.rp-section-title { font-size: 1.1rem; font-weight: 700; color: {navy}; margin: 0; }
.rp-section-sub { color: {slate}; font-size: .9rem; margin: .2rem 0 0; }
.rp-muted { color: {slate}; }
.rp-small { font-size: .82rem; color: {slate}; }
.rp-gold { color: {gold}; }
"""

_CARDS = """
.rp-metric { background: {surface}; border: 1px solid {border}; border-radius: {radius_sm};
             padding: .8rem 1rem; box-shadow: {shadow}; height: 100%; }
.rp-metric-label { color: {slate}; font-size: .8rem; font-weight: 600; display: flex; gap: .35rem;
                   align-items: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rp-metric-label .rp-ms { color: {teal}; }
.rp-metric-value { color: {navy}; font-size: 1.45rem; font-weight: 700; line-height: 1.2; margin-top: .2rem; }
.rp-feature { background: {surface}; border: 1px solid {border}; border-radius: {radius}; padding: 1.1rem 1.15rem;
              box-shadow: {shadow}; height: 100%; transition: border-color .15s ease, box-shadow .15s ease; }
.rp-feature:hover { border-color: {teal_line}; box-shadow: {shadow_lg}; }
.rp-feature-icon { width: 2.3rem; height: 2.3rem; border-radius: {radius_sm}; background: {teal_soft}; color: {teal};
                   display: inline-flex; align-items: center; justify-content: center; font-size: 1.2rem; }
.rp-feature-title { color: {navy}; font-weight: 700; font-size: .98rem; margin: .7rem 0 .25rem; }
.rp-feature-text { color: {slate}; font-size: .86rem; line-height: 1.45; margin: 0; }
"""

_BADGES = """
.rp-badge { display: inline-flex; align-items: center; gap: .4rem; border-radius: 6px; padding: .2rem .55rem;
            font-size: .78rem; font-weight: 600; white-space: nowrap; border: 1px solid transparent; }
.rp-badge::before { content: ""; width: .45rem; height: .45rem; border-radius: 50%; background: currentColor; }
.rp-badge-success { color: {sage}; background: {sage_soft}; border-color: #CFE2D7; }
.rp-badge-info { color: {teal}; background: {teal_soft}; border-color: {teal_line}; }
.rp-badge-warning { color: #9A7224; background: {gold_soft}; border-color: #EBD7AE; }
.rp-badge-danger { color: {red}; background: {red_soft}; border-color: #EBC9C9; }
.rp-badge-neutral { color: {slate}; background: {gray_soft}; border-color: {border}; }
"""

_FLOW = """
.rp-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .65rem; }
.rp-flow-step { position: relative; background: {surface}; border: 1px solid {border}; border-radius: {radius_sm};
                padding: .75rem .8rem .7rem; box-shadow: {shadow}; }
.rp-flow-step.human { border-color: {gold}; }
.rp-flow-step.query, .rp-flow-step.final { background: {teal_soft}; border-color: {teal_line}; }
.rp-flow-num { color: {subtle}; font-size: .7rem; font-weight: 700; letter-spacing: .06em; }
.rp-flow-label { color: {navy}; font-weight: 700; font-size: .9rem; line-height: 1.25; margin: .1rem 0 .35rem; }
.rp-flow-agent { display: inline-flex; align-items: center; gap: .25rem; color: {teal}; background: {teal_soft};
                 border-radius: 5px; padding: .1rem .4rem; font-size: .72rem; font-weight: 600; }
.rp-flow-step.human .rp-flow-agent { color: #8A6420; background: {gold_soft}; }
.rp-flow-step:not(:last-child)::after { content: "→"; position: absolute; right: -.62rem; top: 50%;
                transform: translateY(-50%); color: {teal}; font-weight: 700; font-size: .85rem; z-index: 2; }
@media (max-width: 900px) { .rp-flow-step::after { display: none; } }
"""

_TIMELINE = """
.rp-timeline { list-style: none; margin: 0; padding: 0; }
.rp-tl-item { display: flex; gap: .8rem; position: relative; padding-bottom: .85rem; }
.rp-tl-item:not(:last-child)::after { content: ""; position: absolute; left: .875rem; top: 1.9rem; bottom: .1rem;
                                      width: 2px; background: {border}; }
.rp-tl-item.done:not(:last-child)::after, .rp-tl-item.warning:not(:last-child)::after { background: #BFD8CB; }
.rp-tl-dot { flex: 0 0 1.75rem; height: 1.75rem; border-radius: 50%; display: inline-flex; align-items: center;
             justify-content: center; font-size: .82rem; font-weight: 700; border: 2px solid {border};
             background: {surface}; color: {subtle}; z-index: 1; }
.rp-tl-item.done .rp-tl-dot { background: {sage}; border-color: {sage}; color: #fff; }
.rp-tl-item.warning .rp-tl-dot { background: {gold_soft}; border-color: {gold}; color: #8A6420; }
.rp-tl-item.active .rp-tl-dot { border-color: {teal}; color: {teal}; background: {teal_soft}; }
.rp-tl-item.waiting .rp-tl-dot { border-color: {gold}; color: #8A6420; background: {gold_soft}; }
.rp-tl-item.failed .rp-tl-dot { background: {red_soft}; border-color: {red}; color: {red}; }
.rp-tl-item.skipped .rp-tl-dot { border-style: dashed; }
.rp-tl-body { padding-top: .15rem; min-width: 0; }
.rp-tl-label { color: {navy}; font-weight: 650; font-size: .92rem; line-height: 1.25; }
.rp-tl-item.pending .rp-tl-label, .rp-tl-item.skipped .rp-tl-label { color: {slate}; font-weight: 550; }
.rp-tl-note { color: {slate}; font-size: .78rem; margin-top: .1rem; }
.rp-tl-item.done .rp-tl-note { color: {sage}; }
.rp-tl-item.active .rp-tl-note { color: {teal}; font-weight: 600; }
.rp-tl-item.waiting .rp-tl-note { color: #8A6420; font-weight: 600; }
.rp-tl-item.failed .rp-tl-note { color: {red}; font-weight: 600; }
.rp-spinner { width: .8rem; height: .8rem; border: 2px solid {teal_line}; border-top-color: {teal};
              border-radius: 50%; display: inline-block; animation: rp-spin 1s linear infinite; }
@keyframes rp-spin { to { transform: rotate(360deg); } }
"""

_DOCUMENTS = """
.rp-rcard-title { color: {navy}; font-weight: 700; font-size: 1.02rem; line-height: 1.3; margin: .55rem 0 .25rem;
                  overflow-wrap: anywhere; }
.rp-rcard-desc { color: {slate}; font-size: .86rem; margin: 0 0 .7rem; overflow-wrap: anywhere;
                 display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.rp-rcard-meta { display: flex; flex-wrap: wrap; gap: .9rem; color: {slate}; font-size: .8rem; }
.rp-rcard-meta .rp-ms { color: {teal}; }
.rp-doc-row { display: flex; align-items: center; gap: .85rem; min-width: 0; }
.rp-doc-icon { flex: 0 0 2.4rem; height: 2.6rem; border-radius: 6px; display: inline-flex; align-items: center;
               justify-content: center; font-size: .66rem; font-weight: 800; color: {teal}; background: {teal_soft};
               border: 1px solid {teal_line}; }
.rp-doc-name { color: {navy}; font-weight: 650; overflow-wrap: anywhere; }
.rp-doc-meta { color: {slate}; font-size: .8rem; }
.rp-source { border-left: 3px solid {teal_line}; padding: .15rem 0 .15rem .8rem; margin-bottom: .8rem; }
.rp-source-title { color: {navy}; font-weight: 600; font-size: .92rem; overflow-wrap: anywhere; }
.rp-source-ref { color: {slate}; font-size: .8rem; overflow-wrap: anywhere; }
.rp-source-ref a { color: {teal}; text-decoration: none; }
.rp-meta-row { display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; margin: .7rem 0 1.1rem; }
.rp-meta { color: {slate}; font-size: .82rem; }
.rp-meta b { color: {navy}; font-weight: 650; }
.rp-human { display: inline-flex; align-items: center; gap: .35rem; color: #8A6420; background: {gold_soft};
            border: 1px solid #EBD7AE; border-radius: 6px; padding: .2rem .55rem; font-size: .78rem; font-weight: 650; }
"""

_EMPTY = """
.rp-empty { text-align: center; background: {surface}; border: 1px dashed #D0D5DD; border-radius: {radius};
            padding: 2.4rem 1.5rem 1.2rem; }
.rp-empty-icon { width: 3rem; height: 3rem; border-radius: {radius_sm}; background: {teal_soft}; color: {teal};
                 display: inline-flex; align-items: center; justify-content: center; font-size: 1.4rem; }
.rp-empty h3 { margin: .8rem 0 .3rem; font-size: 1.15rem; color: {navy}; }
.rp-empty p { color: {slate}; margin: 0; }
"""

_WIDGETS = """
/* Keyed containers styled as cards (st.container(key=...)) */
.st-key-composer, [class*="st-key-card"] { background: {surface}; border: 1px solid {border};
    border-radius: {radius}; box-shadow: {shadow}; padding: 1.1rem 1.25rem 1.2rem; }
.st-key-composer { box-shadow: {shadow_lg}; padding: 1.5rem 1.6rem 1.5rem; }
[class*="st-key-tealcard"] { background: {surface}; border: 1px solid {teal_line}; border-top: 3px solid {teal};
    border-radius: {radius}; box-shadow: {shadow_lg}; padding: 1.3rem 1.4rem; }
[class*="st-key-doccard"] { background: {surface}; border: 1px solid {border}; border-radius: {radius};
    box-shadow: {shadow_lg}; padding: 2rem 2.4rem; }
[class*="st-key-doccard"] h1, [class*="st-key-doccard"] h2, [class*="st-key-doccard"] h3 { color: {navy}; }
[class*="st-key-doccard"] h2 { font-size: 1.2rem; margin-top: 1.1rem; padding-bottom: .35rem;
    border-bottom: 1px solid {border}; }
[class*="st-key-doccard"] h3 { font-size: 1.02rem; }
[class*="st-key-doccard"] p, [class*="st-key-doccard"] li { line-height: 1.7; color: #344054; }
.st-key-composer textarea { font-size: 1.02rem; min-height: 120px; }
.stTextArea textarea, .stTextInput input { border-radius: {radius_sm}; }
/* Buttons: primary = teal, secondary = white with border, danger = keyed */
.stButton button, .stDownloadButton button, .stFormSubmitButton button, [data-testid="stPopover"] button {
    border-radius: {radius_sm}; font-weight: 600; min-height: 2.55rem; transition: all .15s ease; }
.stButton button[kind="secondary"], [data-testid="stPopover"] button { border-color: #D0D5DD; color: {navy}; }
.stButton button[kind="secondary"]:hover, [data-testid="stPopover"] button:hover { border-color: {teal};
    color: {teal}; background: #F7FBFB; }
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover { background: {teal_dark};
    border-color: {teal_dark}; }
.stButton button[kind="tertiary"] { color: {slate}; }
[class*="st-key-danger"] button { color: {red}; border-color: #EBC9C9; background: {surface}; }
[class*="st-key-danger"] button:hover { background: {red_soft}; border-color: {red}; color: {red}; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed {teal_line}; background: #FBFDFD;
    border-radius: {radius_sm}; padding: 1.6rem 1.2rem; }
[data-testid="stExpander"] details { border-radius: {radius_sm}; background: {surface}; }
"""

_SIDEBAR = """
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] { margin-top: auto; padding-bottom: 1.25rem; }
[data-testid="stSidebarNav"] a { border-radius: {radius_sm}; border-left: 3px solid transparent; }
[data-testid="stSidebarNav"] a span { font-weight: 550; color: {navy}; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: {teal_soft}; border-left-color: {teal}; }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: {teal}; font-weight: 650; }
[data-testid="stSidebarNav"] a:hover { background: #EEF6F6; }
.rp-sys { display: flex; align-items: center; gap: .55rem; background: {surface}; border: 1px solid {border};
          border-radius: {radius_sm}; padding: .65rem .8rem; color: {navy}; font-weight: 650; font-size: .86rem; }
.rp-sys-dot { width: .55rem; height: .55rem; border-radius: 50%; background: {sage};
              box-shadow: 0 0 0 3px {sage_soft}; flex: 0 0 auto; }
.rp-sys.warn .rp-sys-dot { background: {gold}; box-shadow: 0 0 0 3px {gold_soft}; }
.rp-sys.down .rp-sys-dot { background: {red}; box-shadow: 0 0 0 3px {red_soft}; }
"""

_COMPACT = """
.block-container { padding-top: 1.2rem; }
.rp-feature, .rp-metric { padding: .65rem .8rem; }
.st-key-composer, [class*="st-key-card"], [class*="st-key-tealcard"] { padding: .9rem 1rem; }
"""


def stylesheet(compact: bool = False) -> str:
    parts = [_LAYOUT, _TYPOGRAPHY, _CARDS, _BADGES, _FLOW, _TIMELINE, _DOCUMENTS, _EMPTY, _WIDGETS, _SIDEBAR]
    if compact:
        parts.append(_COMPACT)
    css = "".join(parts)
    for name, value in TOKENS.items():
        css = css.replace("{" + name + "}", value)
    return f"<style>{css}</style>"


def inject_theme(compact: bool = False) -> None:
    st.html(stylesheet(compact))
