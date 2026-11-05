import plotly.express as px
import streamlit as st

from lib import cricket as C, data as D, ui


def render():
    seasons = ui.selected_seasons()
    pt = C.partnership_table()
    pt = pt[pt.season.isin(seasons)]
    c1, c2 = st.columns(2)
    wk = c1.selectbox("Wicket", ["All"] + list(range(1, 11)))
    team = c2.selectbox("Team", ["All"] + D.TEAMS)
    q = pt if wk == "All" else pt[pt.wicket_no == wk]
    if team != "All":
        q = q[q.bat_team == team]
    if q.empty:
        st.warning("No partnerships with these filters.")
        st.stop()
    ui.kpis([("Partnerships", len(q)), ("Hundred stands", int((q.runs >= 100).sum())), ("Highest", int(q.runs.max())),
             ("Average stand", f"{q.runs.mean():.1f}"), ("Average balls", f"{q.balls.mean():.1f}")])
    t1, t2, t3, t4 = st.tabs(["Top stands", "Best pairs", "By wicket", "Opening stands"])

    with t1:
        top = q.sort_values("runs", ascending=False).head(20).assign(date=lambda x: x.date.dt.date)
        ui.table(top[["season", "date", "bat_team", "bowl_team", "pair", "wicket_no", "runs", "balls", "runs_lo", "runs_hi", "ended"]])
        ui.show(px.histogram(q, x="runs", nbins=30, title="Distribution of partnership runs"))

    with t2:
        mn = st.slider("Minimum stands together", 2, 12, 4)
        g = q.groupby(["batter_a", "batter_b"]).agg(stands=("runs", "size"), runs=("runs", "sum"), balls=("balls", "sum"),
                                                    best=("runs", "max"), fifty_plus=("runs", lambda s: (s >= 50).sum())).reset_index()
        g = g[g.stands >= mn]
        g["pair"] = g.batter_a + " & " + g.batter_b
        g["average_stand"] = (g.runs / g.stands).round(1)
        g["runs_per_over"] = (6 * g.runs / g.balls).round(2)
        if g.empty:
            st.info("No pair has batted together that often with these filters.")
        else:
            ui.show(px.bar(g.sort_values("runs", ascending=False).head(15).sort_values("runs"), x="runs", y="pair", orientation="h",
                           color="runs_per_over", color_continuous_scale="YlGn", title="Most productive pairs (colour = runs per over)"))
            ui.table(g.sort_values("runs", ascending=False)[["pair", "stands", "runs", "average_stand", "best", "fifty_plus", "runs_per_over"]])
            st.caption("Pairs are listed alphabetically, not by batting order. Averages include unbroken stands.")

    with t3:
        w = pt if team == "All" else pt[pt.bat_team == team]
        by = w.groupby("wicket_no").agg(stands=("runs", "size"), average=("runs", "mean"), balls=("balls", "mean"),
                                        fifty_share=("runs", lambda s: 100 * (s >= 50).mean())).reset_index()
        ui.show(px.bar(by, x="wicket_no", y="average", text=by.average.round(1), title="Average partnership by wicket"))
        ui.show(px.box(w, x="wicket_no", y="runs", points=False, title="Spread of partnership runs by wicket"))
        ui.table(by.round(1))

    with t4:
        op = pt[pt.wicket_no == 1]
        h = op.groupby(["bat_team", "season"]).runs.mean().reset_index().pivot(index="bat_team", columns="season", values="runs")
        ui.show(px.imshow(h.round(0), text_auto=True, aspect="auto", color_continuous_scale="Greens", title="Average opening stand by team and season"))
        best = op.groupby("pair").agg(stands=("runs", "size"), runs=("runs", "sum"), average=("runs", "mean"), best=("runs", "max")).reset_index()
        ui.table(best[best.stands >= 4].sort_values("average", ascending=False).head(15).round(1))
        st.caption("Opening pairs with at least 4 stands, ranked by average stand.")
