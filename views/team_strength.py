import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from lib import data as D, ml, ui


def render():
    seasons = ui.selected_seasons()
    c1, c2 = st.columns(2)
    k = c1.slider("K-factor (how fast ratings react)", 8, 40, 24, 2)
    reg = c2.slider("Carry-over between seasons (share of rating pulled back to 1500)", 0.0, 0.6, 0.25, 0.05)
    hist, long = ml.elo_table(float(k), float(reg))
    long, hist = long[long.season.isin(seasons)], hist[hist.season.isin(seasons)]

    fig = px.line(long, x="date", y="elo", color="team", color_discrete_map=D.TEAM_COLORS, title="Elo rating after every match")
    fig.add_hline(y=1500, line_dash="dot", line_color="grey")
    ui.show(fig)

    end = long.sort_values("date").groupby(["season", "team"]).elo.last().unstack("season")
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.show(px.imshow(end.round(0), text_auto=True, aspect="auto", color_continuous_scale="RdYlGn",
                          title="Rating at the end of each season"))
    with c2:
        last = long.sort_values("date").groupby("team").tail(1).sort_values("elo", ascending=False)
        peak = long.loc[long.groupby("team").elo.idxmax()][["team", "elo", "date"]].rename(columns={"elo": "peak", "date": "peak date"})
        cur = last[["team", "elo"]].rename(columns={"elo": "latest"}).merge(peak, on="team")
        cur[["latest", "peak"]] = cur[["latest", "peak"]].round(0)
        st.markdown("**Latest and peak rating**")
        ui.table(cur)

    st.markdown("#### Biggest upsets")
    h = hist.copy()
    h["winner_prob"] = np.where(h.y == 1, h.p1, 1 - h.p1)
    up = h.sort_values("winner_prob").head(10)
    up["winner_prob"] = (100 * up.winner_prob).round(0).astype(int).astype(str) + "%"
    ui.table(up[["date", "stage", "winner", "team1", "team2", "winner_prob"]].assign(date=lambda x: x.date.dt.date))

    st.markdown("#### Does Elo predict anything?")
    cal = hist.assign(bin=pd.cut(hist.p1, [0, .4, .45, .5, .55, .6, 1], include_lowest=True)).groupby("bin", observed=True).agg(
        predicted=("p1", "mean"), actual=("y", "mean"), matches=("y", "size")).reset_index(drop=True)
    ui.table(cal.round(2))
    st.caption("In a short, high-variance T20 league the rating is only a weak signal: matches between equal teams are close to coin flips. "
               "The Match predictor page tests this properly out of sample.")

