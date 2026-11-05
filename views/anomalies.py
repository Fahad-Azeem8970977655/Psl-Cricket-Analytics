import plotly.express as px
import streamlit as st
from lib import data as D, ml, ui


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    m = D.load_matches()
    m = m[m.season.isin(seasons)]
    t1, t2 = st.tabs(["Unusual player-seasons", "Unusual matches"])

    with t1:
        c1, c2 = st.columns(2)
        minb = c1.slider("Minimum balls in a season", 60, 400, 120, 10)
        cont = c2.slider("Share flagged", 0.01, 0.15, 0.04, 0.01)
        key = d.assign(_k=d.season.astype(str) + "|" + d.batter)
        ps = D.batting_summary(key, "_k")
        ps[["season", "player"]] = ps._k.str.split("|", n=1, expand=True)
        ps = ps[ps.balls >= minb].copy()
        ps["avg"] = ps.avg.fillna(ps.runs).clip(upper=100)
        if len(ps) < 20:
            st.warning("Too few player-seasons for this filter.")
        else:
            feats = ["sr", "boundary_pct", "dot_pct", "avg", "balls"]
            a = ml.isolation_scores(ps.reset_index(drop=True), feats, cont)
            fig = px.scatter(a, x="sr", y="avg", color="is_anomaly", hover_name="player", hover_data=["season", "runs", "balls"],
                             color_discrete_map={True: "#A8242B", False: "#a9b8a0"}, title="Strike rate vs average (red = flagged)")
            ui.show(fig)
            ui.table(a[a.is_anomaly].sort_values("anomaly_score", ascending=False)[["season", "player", "runs", "balls", "sr", "avg", "boundary_pct", "why"]].round(1))
            st.caption("'Anomaly' means statistically unusual compared with the others, not good or bad.")

    with t2:
        mm = m[m.decided].copy()
        feats = ["inn1_runs", "inn2_runs", "inn1_wkts", "inn2_wkts", "fours", "sixes"]
        a = ml.isolation_scores(mm.reset_index(drop=True), feats, 0.05)
        a["match"] = a.date.dt.date.astype(str) + "  " + a.team1 + " v " + a.team2
        ui.show(px.scatter(a, x="inn1_runs", y="inn2_runs", color="is_anomaly", hover_name="match", size="sixes",
                           color_discrete_map={True: "#A8242B", False: "#a9b8a0"}, title="Both innings totals (red = unusual match)"))
        ui.table(a[a.is_anomaly].sort_values("anomaly_score", ascending=False)[["match", "inn1_runs", "inn2_runs", "fours", "sixes", "winner", "why"]].round(1))


