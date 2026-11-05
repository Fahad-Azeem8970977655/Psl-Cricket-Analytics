import plotly.express as px
import streamlit as st
from lib import data as D, ui, viz


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    bw = D.bowling_summary(d)
    eligible = bw[bw.balls >= 60].sort_values("wkts", ascending=False).bowler.tolist()
    name = st.selectbox("Bowler", eligible)
    df = d[d.bowler == name]
    s = D.bowling_summary(df.assign(_k=name), "_k").iloc[0]
    league = D.bowling_summary(d.assign(_k="x"), "_k").iloc[0]

    ui.kpis([("Wickets", int(s.wkts)), ("Overs", f"{s.overs:.1f}"), ("Economy", f"{s.economy:.2f}", f"League {league.economy:.2f}"),
             ("Balls per wicket", f"{s.bowl_sr:.1f}", f"League {league.bowl_sr:.1f}"), ("Dot %", f"{s.dot_pct:.0f}%", f"League {league.dot_pct:.0f}%"),
             ("Boundary %", f"{s.boundary_pct:.0f}%", f"League {league.boundary_pct:.0f}%")])

    c1, c2 = st.columns([3, 2])
    with c1:
        c = D.run_mix(df)
        ui.show(viz.field_figure(c, title=f"Runs {name} conceded off the bat, by type", height=600))
        st.caption(viz.FIELD_CAPTION)
    with c2:
        ph = df.groupby("phase").agg(runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum"), wk=("bowler_wkt", "sum"),
                                     dots=("is_dot_b", "sum")).reset_index()
        ph["economy"] = ph.runs * 6 / ph.balls
        ph["dot_pct"] = 100 * ph.dots / ph.balls
        ui.show(px.bar(ph, x="phase", y="economy", text=ph.economy.round(2), category_orders={"phase": D.PHASES}, title="Economy by phase"))
        ui.table(ph[["phase", "balls", "wk", "economy", "dot_pct"]].round(1))
        dis = df[df.bowler_wkt].wicket_type.value_counts().rename_axis("how").reset_index(name="wickets")
        if len(dis):
            ui.show(px.pie(dis, names="how", values="wickets", hole=0.45, title="How the wickets came"))

    t1, t2, t3 = st.tabs(["By over", "By season", "Against batters"])
    with t1:
        o = df.groupby("over").agg(runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum"), wk=("bowler_wkt", "sum")).reset_index()
        o["economy"] = o.runs * 6 / o.balls
        ui.show(px.bar(o, x="over", y="balls", color="economy", color_continuous_scale="RdYlGn_r", title="Balls bowled in each over (colour = economy)"))
    with t2:
        x = D.bowling_summary(df, "season")
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(x, x="season", y="wkts", text="wkts", title="Wickets per season"))
        with c2:
            ui.show(px.line(x, x="season", y="economy", markers=True, title="Economy per season"))
    with t3:
        k = df[df.bat_hand != "Unknown"].groupby("bat_hand").agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum"), wk=("bowler_wkt", "sum")).reset_index()
        if len(k):
            k["strike_rate_against"] = 100 * k.runs / k.balls
            ui.table(k.round(1))
        top = df[df.bowler_wkt].player_out.value_counts().head(10).rename_axis("batter dismissed most").reset_index(name="times")
        ui.table(top)

