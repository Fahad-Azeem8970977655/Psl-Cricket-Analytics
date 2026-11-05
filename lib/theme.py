"""Visual identity for the dashboard: the scoreboard and the ground.

Palette   Whites #F6F8F7 (page)  Turf #1F6B45 (primary)  Night #12202F (sidebar, scoreboard)
          Leather #A8242B (sixes, wickets)  Brass #E8B64A (scoreboard digits)
Type      Barlow Condensed for headings and numbers (scoreboards are set in condensed type),
          Source Sans 3 for body text.
"""
import html

import streamlit as st

TURF, TURF_D, NIGHT, LEATHER, BRASS = "#1F6B45", "#164D32", "#12202F", "#A8242B", "#E8B64A"
WHITES, INK, MUTED, LINE = "#F6F8F7", "#17231D", "#5B6B63", "#D9E1DB"
COLORWAY = [TURF, LEATHER, "#C9962E", "#2F5D8A", "#B07D52", "#5A6B7B", "#2E8B8B", "#7A4E8C"]
HEAD_FONT = "Barlow Condensed, Arial Narrow, sans-serif"
BODY_FONT = "Source Sans 3, Segoe UI, system-ui, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.9rem !important; padding-bottom: 3.5rem; max-width: 1280px; }
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stCaptionContainer"], label p, [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] p, button p { font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif; }
.stMarkdown p, .stMarkdown li { line-height: 1.6; }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; color: #17231D; letter-spacing: 0; font-weight: 600 !important; }
h2 { font-size: 1.9rem !important; }
h3 { font-size: 1.5rem !important; margin-top: 1.2rem; }
h4 { font-size: 1.2rem !important; margin-top: .6rem; }
[data-testid="stCaptionContainer"] { color: #5B6B63; }
#MainMenu, footer, [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }

/* page header: title, one line of context, and a crease line (a short thick bar on a thin rule) */
.ph { margin: 0 0 1.4rem 0; padding-bottom: 1rem; border-bottom: 1px solid #D9E1DB; position: relative; }
.ph::after { content: ""; position: absolute; left: 0; bottom: -2px; width: 56px; height: 3px; background: #1F6B45; }
.ph h1 { font-size: 2.7rem !important; line-height: 1.05; margin: 0 0 .35rem 0; padding: 0; }
.ph p { margin: 0; color: #5B6B63; font-size: 1.05rem; max-width: 62ch; }

/* scoreboard strip: all key numbers in one band */
.strip { display: flex; flex-wrap: wrap; background: #12202F; border-radius: 4px; margin: .3rem 0 1.4rem 0; }
.strip .cell { flex: 1 1 150px; padding: .8rem 1.15rem .7rem; border-left: 1px solid rgba(230,237,232,.16); min-width: 0; }
.strip .cell:first-child { border-left: 0; }
.strip .v { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; font-size: 2.15rem; line-height: 1.05;
            color: #E8B64A; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.strip .v.sm { font-size: 1.45rem; padding-top: .35rem; }
.strip .l { margin-top: .25rem; font-size: .84rem; color: #C5D1CA; line-height: 1.25; }

/* match scoreboard */
.board { display: flex; flex-wrap: wrap; background: #12202F; border-radius: 4px; overflow: hidden; margin: .4rem 0 .6rem 0; }
.board .side { flex: 1 1 260px; padding: 1rem 1.4rem 1.05rem; border-top: 5px solid var(--tc, #1F6B45); }
.board .side + .side { border-left: 1px solid rgba(230,237,232,.16); }
.board .tn { color: #E6EDE8; font-size: 1.05rem; font-weight: 600; font-family: 'Source Sans 3', sans-serif; }
.board .sc { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; font-size: 3.6rem; line-height: 1;
             color: #E8B64A; font-variant-numeric: tabular-nums; margin-top: .15rem; }
.board .ov { color: #C5D1CA; font-size: .9rem; margin-top: .15rem; }
.board .res { flex: 1 1 100%; background: #E8B64A; color: #12202F; padding: .55rem 1.4rem; font-weight: 600; font-family: 'Source Sans 3', sans-serif; }
.board-meta { color: #5B6B63; font-size: .92rem; margin: 0 0 1.1rem 0; line-height: 1.5; }
.board-meta span { display: inline-block; margin-right: 1.6rem; }

/* plain metrics elsewhere: flat, numbers set like the scoreboard */
[data-testid="stMetric"] { background: transparent; border-left: 3px solid #D9E1DB; padding: .1rem 0 .1rem .8rem; }
[data-testid="stMetricLabel"] p { font-size: .85rem; color: #5B6B63; }
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; font-size: 2rem; color: #12202F; }

/* tabs: underline, no boxes */
[data-baseweb="tab-list"] { gap: 1.1rem; border-bottom: 1px solid #D9E1DB; }
button[data-baseweb="tab"] { padding: .55rem .15rem; }
button[data-baseweb="tab"] p { font-weight: 600; font-size: 1rem; color: #5B6B63; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #164D32; }
[data-baseweb="tab-highlight"] { background-color: #1F6B45 !important; height: 3px !important; }

/* tables and alerts */
[data-testid="stDataFrame"] { border: 1px solid #D9E1DB; border-radius: 4px; }
[data-testid="stAlert"] { border-radius: 4px; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 0; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(232,182,74,.14); box-shadow: inset 3px 0 0 #E8B64A; }
[data-testid="stSidebarNavSeparator"], [data-testid="stNavSectionHeader"] { font-family: 'Source Sans 3', sans-serif; }
[data-testid="stSidebar"] [data-baseweb="tag"] { background: #E8B64A; }
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #12202F; }

/* hero text on the overview */
.hero-h { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; font-size: 3.2rem; line-height: 1.02;
          color: #12202F; margin: .2rem 0 1rem 0; }
.hero-p { font-size: 1.12rem; line-height: 1.6; color: #17231D; margin: 0 0 .8rem 0; max-width: 46ch; }
.hero-p b { font-weight: 600; color: #164D32; }
.hero-key { margin-top: 1.1rem; padding-top: .8rem; border-top: 1px solid #D9E1DB; font-size: .9rem; color: #5B6B63; max-width: 46ch; }
</style>
"""


def _html(s: str):
    s = " ".join(s.split())            # one line: blank lines or indents would turn into markdown code blocks
    if hasattr(st, "html"):
        st.html(s)
    else:
        st.markdown(s, unsafe_allow_html=True)


def register_plotly():
    """One chart look for every figure. Any problem here falls back to plain Plotly."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        axis = dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE, tickcolor=LINE, ticks="outside",
                    title=dict(font=dict(size=12, color=MUTED)))
        pio.templates["psl"] = go.layout.Template(layout=dict(
            font=dict(family=BODY_FONT, size=13, color=INK),
            title=dict(font=dict(family=HEAD_FONT, size=21, color=INK), x=0, xanchor="left"),
            colorway=COLORWAY, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=axis, yaxis=axis,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, title=dict(side="top"), font=dict(size=12)),
            hoverlabel=dict(bgcolor=NIGHT, font=dict(family=BODY_FONT, color="#E6EDE8")),
            margin=dict(l=10, r=10, t=58, b=10),
            colorscale=dict(sequential=[[0, "#EEF4EF"], [1, TURF_D]], diverging=[[0, LEATHER], [0.5, "#F3F4F1"], [1, TURF]]),
        ))
        pio.templates.default = "plotly_white+psl"
    except Exception:
        pass


def apply():
    _html(CSS)
    register_plotly()
    try:
        from pathlib import Path
        logo = Path(__file__).resolve().parent.parent / "assets" / "logo.svg"
        st.logo(str(logo), icon_image=str(logo), size="large")
    except Exception:
        pass


def page_header(title: str, subtitle: str = ""):
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    _html(f'<div class="ph"><h1>{html.escape(title)}</h1>{sub}</div>')


def strip(items):
    """items: (label, value[, tooltip]). One band of numbers, like a ground scoreboard."""
    cells = []
    for it in items:
        label, value = html.escape(str(it[0])), html.escape(str(it[1]))
        tip = f' title="{html.escape(str(it[2]))}"' if len(it) > 2 and it[2] else ""
        cls = "v sm" if len(str(it[1])) > 12 else "v"
        cells.append(f'<div class="cell"{tip}><div class="{cls}">{value}</div><div class="l">{label}</div></div>')
    _html(f'<div class="strip">{"".join(cells)}</div>')


def scoreboard(sides, result="", meta=()):
    """sides: list of (team, score, overs_text, colour). meta: short lines shown under the board."""
    s = "".join(
        f'<div class="side" style="--tc:{html.escape(c)}"><div class="tn">{html.escape(t)}</div>'
        f'<div class="sc">{html.escape(sc)}</div><div class="ov">{html.escape(ov)}</div></div>' for t, sc, ov, c in sides)
    res = f'<div class="res">{html.escape(result)}</div>' if result else ""
    _html(f'<div class="board">{s}{res}</div>')
    if meta:
        _html('<div class="board-meta">' + "".join(f"<span>{html.escape(x)}</span>" for x in meta if x) + "</div>")


def hero_text(title: str, paragraphs, footnote: str = ""):
    ps = "".join(f'<p class="hero-p">{p}</p>' for p in paragraphs)        # paragraphs may contain <b> (built by us, not data)
    fn = f'<div class="hero-key">{html.escape(footnote)}</div>' if footnote else ""
    _html(f'<div class="hero-h">{html.escape(title)}</div>{ps}{fn}')


def venue_colors(venues):
    """One fixed colour per ground, used on every venue chart."""
    return {v: COLORWAY[i % len(COLORWAY)] for i, v in enumerate(sorted(venues))}
