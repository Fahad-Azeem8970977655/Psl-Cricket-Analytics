import plotly.express as px
import streamlit as st

from lib import cricket as C, ui


def render():
    seasons = tuple(ui.selected_seasons())
    f = C.fielding_tables(seasons)
    p, t = f["players"], f["teams"]
    t1, t2, t3, t4 = st.tabs(["Leaders", "Teams", "Bowler & fielder", "What it can't tell you"])

    with t1:
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Rank by", ["dismissals", "catches", "stumpings", "run_outs", "per_match"])
        mn = c2.slider("Minimum matches listed in a team's XI", 5, 80, 25, 5)
        q = p[p.matches >= mn]
        if q.empty:
            st.info("Nobody has that many matches with these filters.")
        else:
            top = q.sort_values(metric, ascending=False).head(15)
            ui.show(px.bar(top.sort_values(metric), x=metric, y="player", orientation="h", title=f"Fielding leaders by {metric.replace('_', ' ')}"))
            ui.table(q.sort_values(metric, ascending=False)[["player", "matches", "catches", "caught_and_bowled", "stumpings", "run_outs", "direct_hits", "dismissals", "per_match"]])
            keepers = q[q.stumpings >= 3].sort_values("dismissals", ascending=False)
            st.caption("Players with at least 3 stumpings are very likely wicketkeepers: " + (", ".join(keepers.player.head(8)) or "none in this selection")
                       + ". The data does not say who kept wicket, so keepers' catches are mixed with fielders' catches.")

    with t2:
        tt = t.copy()
        long = tt.melt(id_vars=["team", "matches_fielded"], value_vars=["catches", "caught_and_bowled", "stumpings", "run_outs"], var_name="type", value_name="n")
        long["per_match"] = long.n / long.matches_fielded
        ui.show(px.bar(long, x="team", y="per_match", color="type", title="Fielding dismissals per match fielded", color_discrete_sequence=px.colors.sequential.Greens_r))
        ui.table(tt.sort_values("fielding_dismissals_per_match", ascending=False))
        st.caption("Includes dismissals that happen to be taken by the keeper. Bowled and lbw dismissals are not fielding and are shown for context only.")

    with t3:
        c = f["combos"].sort_values("catches", ascending=False).head(15)
        c["pair"] = c.bowler + " / " + c.fielder_1
        ui.show(px.bar(c.sort_values("catches"), x="catches", y="pair", orientation="h", title="Most catches: bowler / catcher"))
        mix = f["mix"]
        ui.show(px.bar(mix, x="season", y="wickets", color="wicket_type", title="How wickets fell, by season"))
        bo = f["by_over"]
        ui.show(px.bar(bo, x="over", y="wickets", color="wicket_type", title="Catches, run-outs and stumpings by over"))

    with t4:
        ui.md(f"""
        - **Only dismissals are recorded.** Dropped catches, misfields, saved runs and throwing accuracy are not in the data, so this is not a full measure of fielding quality.
        - **Substitute fielders** were involved in {f['sub_wickets']} wickets; those are not credited to any listed player.
        - **Run-outs**: {f['run_outs_no_fielder']} of {f['run_outs_total']} run-outs name no fielder in the source, so run-out credit is a lower bound.
        - **Wicketkeepers** cannot be identified directly (the captain / keeper flags are blank), so catches by keepers and fielders are mixed.
        - Per-match rates use the number of matches in which a player was listed in a team's XI.
        """)
