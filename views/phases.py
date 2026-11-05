import numpy as np
import plotly.express as px
import streamlit as st
from lib import data as D, ui


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]

    o = d.groupby(["season", "over"]).agg(runs=("total", "sum"), balls=("bat_balls", "sum"), wk=("is_dismissal", "sum")).reset_index()
    o["run_rate"] = o.runs * 6 / o.balls
    t1, t2, t3, t4 = st.tabs(["League pace", "Teams by phase", "Pace vs spin, left vs right", "Innings 1 vs 2"])

    with t1:
        ui.show(px.imshow(o.pivot(index="season", columns="over", values="run_rate").round(1), text_auto=True, aspect="auto",
                          color_continuous_scale="YlOrRd", title="Runs per over, by season and over"))
        a = d.groupby("over").agg(runs=("total", "sum"), balls=("bat_balls", "sum"), wk=("is_dismissal", "sum"),
                                  b=("is_four", "sum"), s=("is_six", "sum")).reset_index()
        a["run_rate"] = a.runs * 6 / a.balls
        a["wickets_per_100_balls"] = 100 * a.wk / a.balls
        a["boundary_pct"] = 100 * (a.b + a.s) / a.balls
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.line(a, x="over", y="run_rate", markers=True, title="League run rate by over"))
        with c2:
            ui.show(px.bar(a, x="over", y="wickets_per_100_balls", title="Wickets per 100 balls by over"))
        ui.show(px.bar(a, x="over", y="boundary_pct", title="Share of balls hit for four or six, by over"))

    with t2:
        bat = d.groupby(["bat_team", "phase"]).agg(runs=("total", "sum"), balls=("bat_balls", "sum"), wk=("is_dismissal", "sum")).reset_index()
        bat["run_rate"] = bat.runs * 6 / bat.balls
        bat["wkts_per_100"] = 100 * bat.wk / bat.balls
        bowl = d.groupby(["bowl_team", "phase"]).agg(runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum"), wk=("bowler_wkt", "sum")).reset_index()
        bowl["economy"] = bowl.runs * 6 / bowl.balls
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.imshow(bat.pivot(index="bat_team", columns="phase", values="run_rate")[D.PHASES].round(2), text_auto=True,
                              aspect="auto", color_continuous_scale="Greens", title="Batting: runs per over"))
        with c2:
            ui.show(px.imshow(bowl.pivot(index="bowl_team", columns="phase", values="economy")[D.PHASES].round(2), text_auto=True,
                              aspect="auto", color_continuous_scale="Reds_r", title="Bowling: economy (lower is better)"))
        dd = bat[bat.phase == D.PHASES[2]][["bat_team", "run_rate"]].merge(
            bowl[bowl.phase == D.PHASES[2]][["bowl_team", "economy"]], left_on="bat_team", right_on="bowl_team")
        fig = px.scatter(dd, x="run_rate", y="economy", text="bat_team", color="bat_team", color_discrete_map=D.TEAM_COLORS,
                         title="Death overs: batting run rate vs bowling economy (top-left = best)")
        fig.update_yaxes(autorange="reversed")
        ui.show(fig)

    with t3:
        known = d[d.bowl_type != "Unknown"]
        st.caption(f"Bowling-style data covers {100 * len(known) / len(d):.0f}% of deliveries (players in the bio file only).")
        x = known.groupby(["phase", "bowl_type"]).agg(runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum"), wk=("bowler_wkt", "sum")).reset_index()
        x["economy"] = x.runs * 6 / x.balls
        x["balls_per_wicket"] = x.balls / x.wk.replace(0, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(x, x="phase", y="economy", color="bowl_type", barmode="group", category_orders={"phase": D.PHASES},
                           title="Economy: pace vs spin"))
        with c2:
            ui.show(px.bar(x, x="phase", y="balls_per_wicket", color="bowl_type", barmode="group", category_orders={"phase": D.PHASES},
                           title="Balls per wicket: pace vs spin"))
        kh = d[d.bat_hand != "Unknown"]
        y = kh.groupby(["phase", "bat_hand"]).agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum")).reset_index()
        y["strike_rate"] = 100 * y.runs / y.balls
        ui.show(px.bar(y, x="phase", y="strike_rate", color="bat_hand", barmode="group", category_orders={"phase": D.PHASES},
                       title="Strike rate: left- vs right-handed batters"))
        z = known[known.bat_hand != "Unknown"].groupby(["bat_hand", "bowl_type"]).agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum")).reset_index()
        z["strike_rate"] = 100 * z.runs / z.balls
        ui.show(px.bar(z, x="bat_hand", y="strike_rate", color="bowl_type", barmode="group", title="Batter hand vs bowler type (strike rate)"))

    with t4:
        i = d.groupby(["innings", "over"]).agg(runs=("total", "sum"), balls=("bat_balls", "sum")).reset_index()
        i["run_rate"] = i.runs * 6 / i.balls
        i["innings"] = i.innings.map({1: "Batting first", 2: "Chasing"})
        ui.show(px.line(i, x="over", y="run_rate", color="innings", markers=True, title="Run rate by over: batting first vs chasing"))

