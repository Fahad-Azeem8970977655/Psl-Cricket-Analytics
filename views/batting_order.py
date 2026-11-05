import numpy as np
import plotly.express as px
import streamlit as st

from lib import cricket as C, data as D, ui


def render():
    seasons = ui.selected_seasons()
    bo = C.batting_order_table()
    bo = bo[bo.season.isin(seasons)]
    t1, t2, t3, t4 = st.tabs(["By position", "Who bats where", "Roles", "Teams"])

    with t1:
        g = bo.groupby("position").agg(innings=("runs", "size"), runs=("runs", "mean"), balls=("balls", "mean"), out=("out", "mean"),
                                       fours=("fours", "sum"), sixes=("sixes", "sum"), tot_runs=("runs", "sum"), tot_balls=("balls", "sum"),
                                       entry_over=("entry_over", "median"), entry_wkts=("entry_wkts", "mean")).reset_index()
        g["strike_rate"] = 100 * g.tot_runs / g.tot_balls
        g["boundary_pct"] = 100 * (g.fours + g.sixes) / g.tot_balls
        g["dismissed_pct"] = 100 * g.out
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(g, x="position", y="runs", text=g.runs.round(1), title="Average runs by batting position"))
        with c2:
            ui.show(px.line(g, x="position", y="strike_rate", markers=True, title="Strike rate by batting position"))
        ui.table(g[["position", "innings", "runs", "balls", "strike_rate", "boundary_pct", "dismissed_pct", "entry_over", "entry_wkts"]].round(1))
        ui.show(px.box(bo[bo.position <= 8], x="position", y="entry_over", points=False, title="Over in which each position comes in"))
        st.caption("Positions beyond 8 rarely bat; positions come from the order in which batters first appear at the crease.")

    with t2:
        cnt = bo.groupby("player").size()
        pick = st.selectbox("Batter", cnt[cnt >= 15].sort_values(ascending=False).index.tolist())
        x = bo[bo.player == pick]
        pos = x.groupby("position").agg(innings=("runs", "size"), runs=("runs", "sum"), balls=("balls", "sum"), outs=("out", "sum")).reset_index()
        pos["strike_rate"] = (100 * pos.runs / pos.balls).round(1)
        pos["average"] = (pos.runs / pos.outs.replace(0, np.nan)).round(1)
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(pos, x="position", y="innings", title=f"{pick}: innings by batting position"))
        with c2:
            ui.show(px.bar(pos, x="position", y="strike_rate", text="strike_rate", title="Strike rate by position"))
        ui.table(pos)
        by_season = x.groupby("season").position.median().reset_index()
        ui.show(px.line(by_season, x="season", y="position", markers=True, title="Typical position by season (median)").update_yaxes(autorange="reversed"))

    with t3:
        mn = st.slider("Minimum innings", 8, 60, 20)
        r = bo.groupby("player").agg(innings=("runs", "size"), median_position=("position", "median"), median_entry_over=("entry_over", "median"),
                                     runs=("runs", "sum"), balls=("balls", "sum")).reset_index()
        r = r[r.innings >= mn]
        r["strike_rate"] = 100 * r.runs / r.balls
        r["role"] = np.select([r.median_position <= 2, r.median_position <= 4, r.median_position <= 6], ["Opener", "Top order", "Middle order"], "Lower order / finisher")
        if r.empty:
            st.info("No batter has that many innings with these filters.")
        else:
            ui.show(px.scatter(r, x="median_position", y="strike_rate", color="role", size="innings", hover_name="player",
                               title="Where batters bat vs how fast they score"))
            ui.table(r.sort_values("innings", ascending=False)[["player", "role", "median_position", "median_entry_over", "innings", "runs", "strike_rate"]].round(1))

    with t4:
        team = st.selectbox("Team", D.TEAMS)
        x = bo[bo.bat_team == team]
        if x.empty:
            st.info("No innings for this team in the selected seasons.")
        else:
            top = x[x.position <= 6].groupby(["position", "player"]).size().rename("innings").reset_index()
            lead = top.sort_values("innings", ascending=False).groupby("position").head(3).sort_values(["position", "innings"], ascending=[True, False])
            ui.table(lead)
            ph = x.groupby("band").agg(runs=("runs", "sum"), balls=("balls", "sum")).reindex(C.BANDS).reset_index()
            ph["share_of_runs_pct"] = (100 * ph.runs / ph.runs.sum()).round(1)
            ui.show(px.bar(ph, x="band", y="share_of_runs_pct", text="share_of_runs_pct", title=f"{team}: share of bat runs by batting-order band"))
