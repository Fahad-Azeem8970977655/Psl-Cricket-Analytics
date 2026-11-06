"""Wagon wheel: where each batter's runs went, drawn on the ground and pitch.

Data: data/shot_locations.csv (built by build_shot_locations.py from Cricinfo shot tracking).
x, y = where the shot ended, centre = middle of the pitch, 1.0 = boundary rope, bowler's end = top,
off side = left, leg side = right (left-handers are already mirrored).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import data as D
from lib import ui, viz

# Eight equal 45-degree wedges, clockwise from straight back down the ground (leg side is on the right).
SECTORS = ["Mid on", "Mid wicket", "Square leg", "Fine leg", "Third man", "Point", "Cover", "Mid off"]
ORDER = ["Mid off", "Cover", "Point", "Third man", "Fine leg", "Square leg", "Mid wicket", "Mid on"]   # off side, then leg side
OFF_SIDE = {"Mid off", "Cover", "Point", "Third man"}
RUN_STYLE = {1: ("1 run", "#2F5D8A"), 2: ("2 runs", "#17A085"), 3: ("3 runs", "#8E5BB5"),
             4: ("Four", "#E08A00"), 6: ("Six", "#C0262D")}
PHASES = {"All overs": (1, 20), "Powerplay (1-6)": (1, 6), "Middle (7-15)": (7, 15), "Death (16-20)": (16, 20)}
GOOD = "#1F6B45"
GREY = "#8A8F98"


@st.cache_data(show_spinner=False)
def load_shots() -> pd.DataFrame:
    return pd.read_csv(D.DATA / "shot_locations.csv") if hasattr(D, "DATA") else pd.read_csv("data/shot_locations.csv")


@st.cache_data(show_spinner=False)
def _runs_by_batter_season() -> pd.DataFrame:
    d = D.load_deliveries()
    return d.groupby(["batter", "season"], as_index=False).runs_batter.sum()


# ---------------------------------------------------------------------------------- numbers
def sector_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sector (ground order), with share of runs and the run-type split."""
    g = df.groupby("sector").agg(Shots=("runs", "size"), Runs=("runs", "sum")).reindex(ORDER).fillna(0)
    for r, label in [(1, "1s"), (2, "2s"), (3, "3s"), (4, "4s"), (6, "6s")]:
        g[label] = df[df.runs == r].groupby("sector").size().reindex(ORDER).fillna(0)
    total = max(g.Runs.sum(), 1)
    g["% of runs"] = 100 * g.Runs / total
    g.insert(0, "Side", ["Off" if s in OFF_SIDE else "Leg" for s in g.index])
    return g.reset_index().rename(columns={"sector": "Area"})


def summary(df: pd.DataFrame) -> dict:
    runs = int(df.runs.sum())
    if runs == 0:
        return dict(runs=0, shots=0, singles=np.nan, bound=np.nan, off=np.nan, leg=np.nan, v=np.nan)
    sh = lambda m: 100 * df[m].runs.sum() / runs
    return dict(runs=runs, shots=len(df), singles=sh(df.runs == 1), bound=sh(df.runs.isin([4, 6])),
                off=sh(df.sector.isin(OFF_SIDE)), leg=sh(~df.sector.isin(OFF_SIDE)), v=sh(df.sector.isin(["Mid off", "Mid on"])))


# ---------------------------------------------------------------------------------- figure
def _polar(r, deg):
    a = np.radians(deg)
    return r * np.sin(a), r * np.cos(a)


def wheel_figure(df: pd.DataFrame, title: str = "", height: int = 640, small: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="circle", x0=-1.1, y0=-1.1, x1=1.1, y1=1.1, fillcolor="#B7D3A8", line=dict(width=0), layer="below")
    viz._stripes(fig, radius=1.1, bands=14, alpha=0.035)
    fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="#FFFFFF", width=3), layer="below")
    fig.add_shape(type="circle", x0=-0.45, y0=-0.45, x1=0.45, y1=0.45,
                  line=dict(color="rgba(60,90,60,0.55)", width=1.5, dash="dot"), layer="below")
    for k in range(8):                                               # sector dividers
        x0, y0 = _polar(0.16, k * 45); x1, y1 = _polar(1.1, k * 45)
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="rgba(60,90,60,0.35)", width=1), layer="below")
    fig.add_shape(type="rect", x0=-0.024, x1=0.024, y0=-0.155, y1=0.155, fillcolor=viz.PITCH_TAN, line=dict(color="#8c7a47", width=1), layer="above")
    for yy in (-0.115, 0.115):
        fig.add_shape(type="line", x0=-0.04, x1=0.04, y0=yy, y1=yy, line=dict(color="white", width=1.5), layer="above")

    r = np.hypot(df.x, df.y).clip(lower=1e-6)
    scale = np.minimum(1.02 / r, 1.0)                                 # boundaries are drawn to the rope
    X, Y = (df.x * scale).to_numpy(), (df.y * scale).to_numpy()
    n = max(len(df), 1)
    alpha = float(np.clip(260 / n, 0.16, 0.65))               # fewer shots -> stronger lines
    msize = (3.5 if n > 900 else 5) - (0.5 if small else 0)
    for runs, (label, colour) in RUN_STYLE.items():
        m = (df.runs == runs).to_numpy()
        if not m.any():
            continue
        xs, ys = X[m], Y[m]
        lx = np.column_stack([np.zeros(m.sum()), xs, np.full(m.sum(), np.nan)]).ravel()
        ly = np.column_stack([np.zeros(m.sum()), ys, np.full(m.sum(), np.nan)]).ravel()
        width = 0.9 if runs < 4 else 1.2
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=colour, width=width), opacity=alpha,
                                 name=label, legendgroup=label, showlegend=False, hoverinfo="skip"))
        sub = df[m]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", marker=dict(color=colour, size=msize, line=dict(color="white", width=0.4)),
            name=label, legendgroup=label, showlegend=True,
            text=[f"<b>{b}</b> {int(ru)} run{'s' if ru != 1 else ''}<br>{(sp + ' - ') if isinstance(sp, str) else ''}{sec}<br>"
                  f"off {bo} - {se} over {int(ov)}" for b, ru, sp, sec, bo, se, ov in
                  zip(sub.batter, sub.runs, sub.shot_played, sub.sector, sub.bowler, sub.season, sub.over)],
            hovertemplate="%{text}<extra></extra>"))

    t = sector_table(df).set_index("Area")
    fs = 11 if small else 13
    for k, name in enumerate(SECTORS):
        x, y = _polar(1.13, k * 45 + 22.5)
        xa = "left" if x > 0.3 else "right" if x < -0.3 else "center"
        ya = "bottom" if y > 0.3 else "top" if y < -0.3 else "middle"
        fig.add_annotation(x=x, y=y, xanchor=xa, yanchor=ya, showarrow=False, align=xa if xa != "center" else "center",
                           text=f"{name}<br><b style='font-size:{fs + 5}px'>{t.loc[name, '% of runs']:.0f}%</b>",
                           font=dict(size=fs, color=viz.INK))
    fig.add_annotation(x=-1.6, y=1.46, text="\u25c0 Off side", showarrow=False, xanchor="left", font=dict(size=fs, color=GREY))
    fig.add_annotation(x=1.6, y=1.46, text="Leg side \u25b6", showarrow=False, xanchor="right", font=dict(size=fs, color=GREY))
    fig.add_annotation(x=0, y=1.46, text="Bowler's end \u25b2", showarrow=False, font=dict(size=fs, color=GREY))
    fig.update_xaxes(visible=False, range=[-1.64, 1.64], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-1.4, 1.56], scaleanchor="x", fixedrange=True)
    fig.update_layout(height=height, title=dict(text=title, x=0.01, font=dict(family=viz.HEAD_FONT, size=22 if not small else 18)),
                      margin=dict(l=0, r=0, t=50 if title else 10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=0.0, x=0.5, xanchor="center", yanchor="top", font=dict(size=12), itemsizing="constant"), hoverlabel=dict(font_size=12))
    return fig


# ---------------------------------------------------------------------------------- page
def _filters(shots: pd.DataFrame):
    seasons_all = sorted(int(s) for s in shots.season.unique())
    chosen = [s for s in ui.selected_seasons() if s in seasons_all]
    c = st.columns([1.2, 1.1, 1.3, 1.1])
    inn = c[0].radio("Innings", ["Both", "1st", "2nd"], horizontal=True, key="ww_inn")
    phase = c[1].selectbox("Phase", list(PHASES), key="ww_phase")
    bt = c[2].radio("Bowler type", ["All", "Pace", "Spin"], horizontal=True, key="ww_bt")
    runs = c[3].multiselect("Run types", [1, 2, 3, 4, 6], default=[1, 2, 3, 4, 6], key="ww_runs",
                            format_func=lambda r: RUN_STYLE[r][0])
    d = shots[shots.season.isin(chosen) & shots.runs.isin(runs or [1, 2, 3, 4, 6])]
    if inn != "Both":
        d = d[d.innings == (1 if inn == "1st" else 2)]
    lo, hi = PHASES[phase]
    d = d[d.over.between(lo, hi)]
    if bt != "All":
        d = d[d.bowl_type == bt]
    return d, chosen


def render():
    shots = load_shots()
    pool, chosen = _filters(shots)
    if not chosen:
        st.warning("No shot-location data for the seasons selected in the sidebar. Tracking exists for 2018-2026 only.")
        return
    if 2019 in chosen:
        st.info("2019 has shot locations for only about 13% of balls, so it will thin out the wheel. "
                "2016 and 2017 have none. Every other season is 80-85% tracked.")

    strength = pool.groupby("batter").runs.sum().sort_values(ascending=False)
    names = strength[strength >= 20].index.tolist()
    if not names:
        st.warning("Too few shots match these filters. Widen the filters.")
        return
    top = st.columns([1.6, 1.4])
    default = names.index("Babar Azam") if "Babar Azam" in names else 0
    player = top[0].selectbox("Batter", names, index=default, key="ww_player",
                              format_func=lambda n: f"{n}  ({int(strength[n])} runs tracked)")
    mode = top[1].radio("Compare with", ["Nobody", "League average", "Another batter"], horizontal=True, key="ww_cmp")
    other = None
    if mode == "Another batter":
        rest = [n for n in names if n != player]
        other = st.selectbox("Second batter", rest, key="ww_player2", format_func=lambda n: f"{n}  ({int(strength[n])} runs tracked)")

    p1 = pool[pool.batter == player]
    s1 = summary(p1)
    seasons_txt = f"{min(chosen)}-{max(chosen)}" if len(chosen) > 1 else str(chosen[0])
    tracked_all = shots[(shots.batter == player) & shots.season.isin(chosen)].runs.sum()
    actual = _runs_by_batter_season()
    actual = actual[(actual.batter == player) & actual.season.isin(chosen)].runs_batter.sum()
    cov = 100 * tracked_all / actual if actual else np.nan
    ui.kpis([("Runs on the wheel", f"{s1['runs']:,}", "Runs from the shots drawn below, after the filters"),
             ("Shots drawn", f"{s1['shots']:,}"),
             ("Singles share of runs", ui.fmt_pct(s1["singles"], 0)),
             ("Boundary share of runs", ui.fmt_pct(s1["bound"], 0)),
             ("Off side", ui.fmt_pct(s1["off"], 0)), ("Leg side", ui.fmt_pct(s1["leg"], 0)),
             ("Through the V", ui.fmt_pct(s1["v"], 0), "Mid off + mid on wedges, straight down the ground"),
             ("Tracked / actual runs", ui.fmt_pct(cov, 0), "Share of this batter's runs in the chosen seasons that have a shot location")])

    if other:
        p2 = pool[pool.batter == other]
        a, b = st.columns(2)
        with a:
            ui.show(wheel_figure(p1, f"{player}  ({seasons_txt})", height=560, small=True), key="ww_a")
        with b:
            ui.show(wheel_figure(p2, f"{other}  ({seasons_txt})", height=560, small=True), key="ww_b")
        ref, ref_name = p2, other
    else:
        left, right = st.columns([3, 2])
        with left:
            ui.show(wheel_figure(p1, f"{player}  ({seasons_txt})"), key="ww_a")
        ref, ref_name = (pool, "League average") if mode == "League average" else (None, None)
        with right:
            t1 = sector_table(p1)
            bar = pd.DataFrame({"Area": t1.Area, player: t1["% of runs"]})
            if ref is not None:
                bar[ref_name] = sector_table(ref)["% of runs"].values
            long = bar.melt("Area", var_name="Who", value_name="pct")
            fig = px.bar(long, x="pct", y="Area", color="Who", barmode="group", orientation="h",
                         category_orders={"Area": ORDER[::-1]}, text=long.pct.round(0).astype(int).astype(str) + "%",
                         color_discrete_sequence=[GOOD, GREY], title="Share of runs by area")
            fig.update_xaxes(title="% of runs")
            fig.update_yaxes(title="")
            ui.show(fig, key="ww_bar")

    t = sector_table(p1)
    show = t.rename(columns={"% of runs": f"{player} % of runs"})
    if ref is not None:
        tr = sector_table(ref)
        show[f"{ref_name} % of runs"] = tr["% of runs"].values
        show["Difference (pts)"] = (t["% of runs"].values - tr["% of runs"].values)
    num = show.select_dtypes("number").columns
    show[num] = show[num].round(1)
    for col in ["Shots", "Runs", "1s", "2s", "3s", "4s", "6s"]:
        show[col] = show[col].astype(int)
    st.markdown("**Runs by area of the ground**")
    ui.table(show)

    with st.expander("How to read this, and what the data can and cannot say"):
        ui.md(f"""
        - **Each line is one scoring shot**, drawn from the middle of the pitch to where Cricinfo recorded it ending. Boundaries are drawn to the rope.
          Hover a dot to see the bowler, season, over and shot. Bowler's end is at the top, **off side on the left, leg side on the right**.
        - **Percentages are the share of the batter's runs** that went through each of the 8 equal wedges. They add to 100%.
        - **Left-handers are mirrored** so the off side is always on the left. Handedness is read from each batter's own off-side shots
          (it agrees with your bio file for 96% of batters).
        - **Source:** Cricinfo shot tracking, from *The Ultimate Ball-by-Ball Cricket Dataset* (Ari Adaikalam, Kaggle), matched to your Cricsheet balls
          (99.9% same batter and bowler). No location is estimated from runs.
        - **Coverage:** 2016 and 2017 have no locations. 2019 has about 13%. 2018 and 2020-2026 are 80-85% of all balls, and about 99% of balls
          that scored runs. Only about 57% of dot balls are tracked, so **dot-ball percentages are not shown**: they would be wrong.
        - Total coverage for {player} in the chosen seasons is shown as *Tracked / actual runs* above. Below about 90%, treat the percentages as approximate.
        - Shot type (drive, pull...) is also in the data but is not used here yet.
        """)
