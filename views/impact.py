import plotly.express as px
import streamlit as st
from lib import ml, ui


def render():
    seasons = ui.selected_seasons()
    c1, c2, c3 = st.columns(3)
    wv = c1.slider("Value of a wicket (runs)", 8, 30, 18)
    min_bat = c2.slider("Minimum balls faced", 60, 1000, 300, 20)
    min_bowl = c3.slider("Minimum balls bowled", 60, 1000, 240, 20)
    bat, bowl, allp = ml.impact_tables(tuple(seasons), float(wv))
    bat, bowl = bat[bat.balls >= min_bat], bowl[bowl.balls >= min_bowl]

    t1, t2, t3, t4 = st.tabs(["Batting", "Bowling", "Combined", "Method"])
    with t1:
        top = bat.sort_values("bat_impact", ascending=False).head(15)
        ui.show(px.bar(top.sort_values("bat_impact"), x="bat_impact", y="batter", orientation="h", title="Batting impact (runs above average, net of dismissals)"))
        fig = px.scatter(bat, x="balls", y="bat_impact_per100", hover_name="batter", size=bat.runs.clip(lower=1), title="Impact per 100 balls vs balls faced")
        fig.add_hline(y=0, line_dash="dot")
        ui.show(fig)
        ui.table(bat.sort_values("bat_impact", ascending=False)[["batter", "innings", "runs", "balls", "raa_runs", "outs", "bat_impact", "bat_impact_per100"]].round(1))
    with t2:
        top = bowl.sort_values("bowl_impact", ascending=False).head(15)
        ui.show(px.bar(top.sort_values("bowl_impact"), x="bowl_impact", y="bowler", orientation="h", title="Bowling impact (runs saved + wickets above average)"))
        fig = px.scatter(bowl, x="balls", y="bowl_impact_per100", hover_name="bowler", size=bowl.wkts.clip(lower=1), title="Impact per 100 balls vs balls bowled")
        fig.add_hline(y=0, line_dash="dot")
        ui.show(fig)
        ui.table(bowl.sort_values("bowl_impact", ascending=False)[["bowler", "matches", "balls", "wkts", "saved", "w_above", "bowl_impact", "bowl_impact_per100"]].round(1))
    with t3:
        a = allp[(allp.bat_balls >= min_bat) | (allp.bowl_balls >= min_bowl)].sort_values("total_impact", ascending=False)
        ui.show(px.bar(a.head(15).sort_values("total_impact"), x=["bat_impact", "bowl_impact"], y="player", orientation="h",
                       title="Total impact = batting + bowling", color_discrete_sequence=["#1F6B45", "#A8242B"]))
        both = a[(a.bat_balls >= min_bat / 2) & (a.bowl_balls >= min_bowl / 2)]
        if len(both):
            st.markdown("**All-rounders** (meaningful batting and bowling)")
            ui.table(both[["player", "bat_balls", "bowl_balls", "bat_impact", "bowl_impact", "total_impact"]].round(1))
    with t4:
        ui.md(f"""
    **Batting impact** = sum of (runs scored - league runs per ball in that season/innings/over x balls faced) - {wv} x (dismissals - expected dismissals).

    **Bowling impact** = runs saved against the league average for the same context + {wv} x (wickets - expected wickets).

    - Comparing against the same season, innings and over removes the effect of scoring getting faster over the years and of death overs being easier to score in.
    - It rewards volume as well as quality, so the per-100-balls columns show efficiency.
    - It does not know the batting position, the opposition, the pitch or who was bowling at the other end.
    - The wicket value is a judgement; move the slider to see how rankings change.
    """)
