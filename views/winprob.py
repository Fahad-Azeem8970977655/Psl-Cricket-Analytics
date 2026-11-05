import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import data as D, ml, ui


def render():
    seasons = ui.selected_seasons()
    res = ml.inmatch_models()
    s = ml.ball_states()
    m = D.load_matches()
    t1, t2, t3, t4 = st.tabs(["Match explorer", "What-if", "Model quality", "Win probability added"])

    with t1:
        mm = m[m.season.isin(seasons) & m.decided & (m.result_type == "normal")].sort_values("date", ascending=False)
        label = mm.apply(lambda r: f"{r.date:%Y-%m-%d}  {r.team1} v {r.team2}  ({r.stage})", axis=1)
        pick = st.selectbox("Match", label.tolist())
        r = mm.loc[label[label == pick].index[0]]
        ui.kpis([("Batting first", r.bat_first), ("First innings", f"{int(r.inn1_runs)}/{int(r.inn1_wkts)}"),
                 ("Second innings", f"{int(r.inn2_runs)}/{int(r.inn2_wkts)}"), ("Winner", r.winner)])
        c = ml.chase_wpa()
        c = c[c.match_idx == r.match_idx].reset_index(drop=True)
        c["ball_of_innings"] = np.arange(1, len(c) + 1)
        chaser = r.chase_team
        fig = go.Figure(go.Scatter(x=c.ball_of_innings, y=100 * c.wp, mode="lines", name=f"{chaser} win %", line=dict(color=D.TEAM_COLORS.get(chaser, "#1F6B45"), width=3)))
        w = c[c.is_dismissal]
        fig.add_trace(go.Scatter(x=w.ball_of_innings, y=100 * w.wp, mode="markers", name="wicket", marker=dict(color="#A8242B", size=9)))
        fig.add_hline(y=50, line_dash="dot")
        fig.update_layout(title=f"Chasing win probability: {chaser}", yaxis_title="win probability %", xaxis_title="delivery of the chase", height=380)
        ui.show(fig)
        st.markdown("**Biggest swings**")
        sw = c.reindex(c.dwp.abs().sort_values(ascending=False).index).head(8)
        sw = sw.assign(swing_pts=(100 * sw.dwp).round(1), event=np.where(sw.is_dismissal, "wicket (" + sw.wicket_type.fillna("") + ")", sw.runs_batter.astype(str) + " off the bat"))
        ui.table(sw[["over", "ball", "batter", "bowler", "event", "swing_pts"]])
        p1 = s[(s.match_idx == r.match_idx) & (s.innings == 1) & (s.cum_balls >= 6)].copy()
        if len(p1):
            p1["projected"] = res["proj_model"].predict(p1[ml.PR_FEATS])
            p1["ball_of_innings"] = np.arange(6, 6 + len(p1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p1.ball_of_innings, y=p1.projected, name="ML projection", line=dict(color="#1F6B45", width=3)))
            fig.add_trace(go.Scatter(x=p1.ball_of_innings, y=p1.crr * 20, name="Run rate x 20", line=dict(color="#C9962E", dash="dash")))
            fig.add_hline(y=r.inn1_runs, line_dash="dot", annotation_text=f"final {int(r.inn1_runs)}")
            fig.update_layout(title=f"First-innings projection: {r.bat_first}", yaxis_title="projected total", xaxis_title="delivery of the innings", height=360)
            ui.show(fig)

    with t2:
        c1, c2, c3, c4 = st.columns(4)
        need = c1.number_input("Runs needed", 1, 250, 60)
        left = c2.number_input("Balls left", 1, 120, 36)
        wk = c3.number_input("Wickets in hand", 0, 10, 6)
        crr = c4.number_input("Current run rate", 0.0, 20.0, 8.0)
        rows = [{"model": n, "chasing team wins": f"{100 * ml.chase_prob(res, need, left, wk, crr, n):.0f}%"} for n in res["chase_models"]]
        ui.table(pd.DataFrame(rows))
        st.metric("Required run rate", f"{need * 6 / left:.2f}")
        sc = np.arange(40, 181, 10)
        curve = pd.DataFrame({"runs needed": sc, "win %": [100 * ml.chase_prob(res, x, left, wk, crr) for x in sc]})
        ui.show(px.line(curve, x="runs needed", y="win %", markers=True, title=f"With {left} balls and {wk} wickets left"))

    with t3:
        st.markdown("**Chase win probability** (5-fold cross-validation, folds split by match)")
        ui.table(res["chase_eval"].round(3))
        st.markdown("Accuracy by stage of the chase")
        ui.table(res["chase_bucket"].round(3).reset_index().rename(columns={"index": "stage"}))
        st.markdown("**First-innings projection** (mean absolute error in runs, lower is better)")
        ui.table(res["proj_eval"].round(1).reset_index())
        ui.kpis([("ML model error", f"{res['proj_mae'][0]:.1f} runs"), ("Run-rate x 20 error", f"{res['proj_mae'][1]:.1f} runs")])
        st.caption("Win probability is a model of the match state (runs, balls, wickets), not of team quality or the pitch.")

    with t4:
        w = ml.chase_wpa()
        w = w[w.season.isin(seasons)]
        mn = st.slider("Minimum balls", 30, 300, 100, 10)
        bat = w.groupby("batter").agg(wpa=("dwp", "sum"), balls=("bat_balls", "sum")).reset_index()
        bowl = w.assign(neg=-w.dwp).groupby("bowler").agg(wpa=("neg", "sum"), balls=("bowl_balls", "sum")).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            b = bat[bat.balls >= mn].sort_values("wpa", ascending=False).head(15)
            ui.show(px.bar(b.sort_values("wpa"), x="wpa", y="batter", orientation="h", title="Batters: win probability added while chasing"))
        with c2:
            b = bowl[bowl.balls >= mn].sort_values("wpa", ascending=False).head(15)
            ui.show(px.bar(b.sort_values("wpa"), x="wpa", y="bowler", orientation="h", title="Bowlers: win probability added defending"))
        st.caption("1.0 = one full win's worth of probability. Only second innings are counted, so this measures pressure performance in chases.")

