import pandas as pd
import plotly.express as px
import streamlit as st
from lib import data as D, theme, ui


def render():
    seasons = ui.selected_seasons()
    m = D.load_matches()
    m = m[m.season.isin(seasons)]
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]

    vc = theme.venue_colors(D.load_matches().venue.unique())
    dec = m[m.decided]
    v = m.groupby("venue").agg(matches=("match_idx", "size"), avg_first_innings=("inn1_runs", "mean"),
                               avg_second_innings=("inn2_runs", "mean"), fours=("fours", "mean"), sixes=("sixes", "mean"),
                               avg_total=("total_runs", "mean")).reset_index()
    v = v.merge(dec.groupby("venue").chase_won.mean().mul(100).rename("chase_win_pct").reset_index(), on="venue", how="left")
    ui.table(v.round(1).sort_values("matches", ascending=False))

    c1, c2 = st.columns(2)
    with c1:
        ui.show(px.bar(v.sort_values("avg_first_innings"), x="avg_first_innings", y="venue", color="venue", color_discrete_map=vc,
                       orientation="h", title="Average first-innings score"))
    with c2:
        fig = px.scatter(v, x="avg_first_innings", y="chase_win_pct", size="matches", color="venue", color_discrete_map=vc,
                         hover_name="venue", size_max=36, title="High-scoring grounds vs chasing success",
                         labels={"avg_first_innings": "average first-innings score", "chase_win_pct": "chasing team won (%)"})
        fig.add_hline(y=50, line_dash="dot")
        ui.show(fig)

    fig = px.box(m, x="venue", y="inn1_runs", color="venue", color_discrete_map=vc, points="all", title="Spread of first-innings totals")
    fig.update_layout(showlegend=False)
    ui.show(fig)

    t1, t2, t3 = st.tabs(["Run rate by phase", "Season trend", "Where each team played"])
    with t1:
        ph = d.groupby(["venue", "phase"]).agg(runs=("total", "sum"), balls=("bat_balls", "sum")).reset_index()
        ph["run_rate"] = ph.runs * 6 / ph.balls
        piv = ph.pivot(index="venue", columns="phase", values="run_rate")[D.PHASES]
        ui.show(px.imshow(piv.round(2), text_auto=True, aspect="auto", color_continuous_scale="YlOrRd", title="Runs per over by venue and phase"))
    with t2:
        s = m.groupby(["season", "venue"]).inn1_runs.mean().reset_index()
        ui.show(px.line(s, x="season", y="inn1_runs", color="venue", color_discrete_map=vc, markers=True, title="Average first-innings score by season and venue"))
    with t3:
        cnt = pd.concat([m[["team1", "venue"]].rename(columns={"team1": "team"}), m[["team2", "venue"]].rename(columns={"team2": "team"})])
        piv = cnt.groupby(["team", "venue"]).size().unstack(fill_value=0)
        ui.show(px.imshow(piv, text_auto=True, aspect="auto", color_continuous_scale="Greens", title="Matches played by team and venue"))
        st.caption("Franchise 'home grounds' were mostly not used (e.g. Quetta and Peshawar never hosted), so real home advantage cannot be read from the nominal ground.")

