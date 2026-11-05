import numpy as np
import plotly.express as px
import streamlit as st
from lib import data as D, ui


def render():
    seasons = ui.selected_seasons()
    fr, pt, tf, aw = D.load_final_results(), D.load_points(), D.load_third_fourth(), D.load_awards()
    m = D.load_matches()
    fr, pt = fr[fr.season.isin(seasons)], pt[pt.season.isin(seasons)]

    t1, t2, t3, t4 = st.tabs(["Finals", "League tables", "3rd / 4th place", "Awards check"])

    with t1:
        f = fr.dropna(subset=["margin_value"])
        fig = px.bar(f, x="season", y="margin_value", color="margin_type", text="champion",
                     color_discrete_sequence=["#1F6B45", "#A8242B"], title="Winning margin in each final (runs or wickets)")
        ui.show(fig)
        fm = m[(m.stage == "final") & m.season.isin(seasons)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Finals won chasing", f"{int(fm.chase_won.sum())} of {len(fm)}")
        c2.metric("Finals won batting first", f"{int(fm.bat_first_won.sum())} of {len(fm)}")
        c3.metric("Closest final", fr.loc[fr.margin_value.idxmin(), "season"] if fr.margin_value.notna().any() else "-")
        ui.table(fr[["season", "champion", "runner_up", "final_venue", "final_date", "margin"]])

    with t2:
        piv = pt.pivot(index="team", columns="season", values="points")
        fig = px.imshow(piv, text_auto=True, aspect="auto", color_continuous_scale="Greens", title="League points by team and season")
        ui.show(fig)
        champ = fr[["season", "champion"]].merge(pt, left_on=["season", "champion"], right_on=["season", "team"], how="left")
        champ = champ.rename(columns={"final_position": "league_position"})
        c1, c2 = st.columns(2)
        c1.metric("Champions that topped the league", f"{int((champ.league_position == 1).sum())} of {len(champ)}")
        c2.metric("Champions that finished 4th", f"{int((champ.league_position == 4).sum())} of {len(champ)}")
        fig = px.bar(champ, x="season", y="league_position", text="champion", title="League-stage position of each champion (1 = topped the table)")
        fig.update_yaxes(autorange="reversed")
        ui.show(fig)
        fig = px.scatter(pt, x="points", y="net_run_rate", color="team", color_discrete_map=D.TEAM_COLORS, hover_data=["season"],
                         title="Points vs net run rate (every team-season)")
        ui.show(fig)

    with t3:
        st.caption("Two definitions exist: *playoff* = the teams eliminated in the qualifier/eliminator, *league* = 3rd/4th by points. "
                   "They differ in many seasons, so label which one you use.")
        x = tf[tf.season.isin(seasons)].copy()
        x["same?"] = np.where((x.third_place_playoff == x.third_place_league_table) &
                              (x.fourth_place_playoff == x.fourth_place_league_table), "same", "different")
        ui.table(x.drop(columns=["source_url", "notes"], errors="ignore"))

    with t4:
        a = aw[aw.season.isin(seasons)]
        ui.table(a[["season", "green_cap_player", "green_cap_runs", "maroon_cap_player", "maroon_cap_wickets"]])
        c1, c2 = st.columns(2)
        ui.show(px.bar(a, x="season", y="green_cap_runs", text="green_cap_player", title="Most runs in a season (Green Cap)"))
        ui.show(px.bar(a, x="season", y="maroon_cap_wickets", text="maroon_cap_player", title="Most wickets in a season (Maroon Cap)"))
        st.caption("The 2022 total of 588 is correct once super-over runs are excluded (see Overview > Reconciliation).")

