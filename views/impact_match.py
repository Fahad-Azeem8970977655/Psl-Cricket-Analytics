"""Match-level impact score: batting + bowling + fielding + pressure + win bonus."""
import pandas as pd
import plotly.express as px
import streamlit as st

from lib import data as D
from lib import ui
from lib.impact_match import Weights, build_player_impact, potm_validation
from lib.runout_fix import load_fixes

COMP = ["batting_impact", "bowling_impact", "fielding_impact", "pressure_impact", "win_bonus"]
COMP_COLORS = {"batting_impact": "#1F6B45", "bowling_impact": "#A8242B", "fielding_impact": "#2F5D8A",
               "pressure_impact": "#B8860B", "win_bonus": "#8A8F98"}


@st.cache_data(show_spinner="Building match-level impact scores...")
def _table(wv, catch, stump, runout, pressure, win_bonus) -> pd.DataFrame:
    w = Weights(wicket_value=wv, catch=catch, stumping=stump, run_out=runout, pressure=pressure, win_bonus=win_bonus)
    d, xi = D.load_deliveries(), D.load_xi()
    ov, kp = load_fixes(d, xi, D.load_bio())
    return build_player_impact(d, D.load_matches(), xi, w, runout_overrides=ov, keepers=kp)


def _export(x: pd.DataFrame) -> str:
    f = x.copy()
    f["date"] = f.date.dt.strftime("%Y-%m-%d")
    return f.round(3).to_csv(index=False)


def render():
    seasons = ui.selected_seasons()
    with st.expander("Weights (all in runs)", expanded=False):
        c = st.columns(6)
        wv = c[0].slider("Wicket", 8, 30, 18, key="pi_wv")
        catch = c[1].slider("Catch", 0, 15, 5, key="pi_catch")
        stump = c[2].slider("Stumping", 0, 20, 8, key="pi_st")
        runout = c[3].slider("Run-out", 0, 20, 10, key="pi_ro")
        pressure = c[4].slider("Pressure extra", 0.0, 1.0, 0.25, 0.05, key="pi_pr")
        wb = c[5].slider("Win bonus", 0.0, 10.0, 3.0, 0.5, key="pi_wb")
    full = _table(float(wv), float(catch), float(stump), float(runout), float(pressure), float(wb))
    df = full[full.season.isin(seasons)]
    val = potm_validation(df)
    ui.kpis([("Player-matches", f"{len(df):,}"), ("Matches", f"{df.match_id.nunique():,}"),
             ("Player of the match = top score", ui.fmt_pct(val["top1"]), "How often the official Player of the Match has the highest impact score in that match"),
             ("Player of the match in top 3", ui.fmt_pct(val["top3"])), ("Average POTM rank", f"{val['avg_rank']:.1f}")])

    def leaderboard():
        min_m = st.slider("Minimum matches", 1, 100, 30, key="pi_minm")
        by = st.radio("Rank by", ["Average per match", "Total"], horizontal=True, key="pi_by")
        g = df.groupby("player").agg(matches=("match_id", "nunique"), **{c: (c, "sum") for c in COMP}, total=("impact_score", "sum")).reset_index()
        g["per_match"] = g.total / g.matches
        g = g[g.matches >= min_m].sort_values("per_match" if by.startswith("Average") else "total", ascending=False)
        top = g.head(15)
        long = top.melt(id_vars=["player"], value_vars=COMP, var_name="component", value_name="runs")
        long["component"] = long.component.str.replace("_impact", "").str.replace("_", " ")
        order = top.sort_values("per_match" if by.startswith("Average") else "total").player.tolist()
        # components are shown as totals so the stack adds up to the total
        fig = px.bar(long, x="runs", y="player", color="component", orientation="h", category_orders={"player": order},
                     color_discrete_sequence=list(COMP_COLORS.values()), title="Top 15 by impact score, split into components (total runs)")
        ui.show(fig)
        ui.table(g[["player", "matches", "per_match", "total"] + COMP].round(1))

    def match_view():
        mm = D.load_matches()
        mm = mm[mm.season.isin(seasons)].sort_values("date", ascending=False)
        lab = {r.match_id: f"{r.date:%d %b %Y}  {r.team1} v {r.team2}  ({r.stage})" for r in mm.itertuples()}
        mid = st.selectbox("Match", list(lab), format_func=lab.get, key="pi_match")
        x = full[full.match_id == mid].sort_values("impact_score", ascending=False)
        win = D.load_matches().set_index("match_id").loc[mid, "winner"]
        st.caption(f"Winner: **{win if isinstance(win, str) else 'no result'}**.  ★ marks the official Player of the Match.")
        x = x.assign(player=x.player + x.player_of_match.map({1: " ★", 0: ""}))
        ui.show(px.bar(x.head(12).sort_values("impact_score"), x=COMP, y="player", orientation="h",
                       color_discrete_sequence=list(COMP_COLORS.values()), title="Top 12 in this match"))
        ui.table(x[["player", "team", "runs", "balls_faced", "wickets", "runs_conceded", "catches", "stumpings", "run_outs",
                    "batting_impact", "bowling_impact", "fielding_impact", "pressure_impact", "win_bonus", "impact_score"]].round(1))

    def player_view():
        names = sorted(df.player.unique())
        default = "Babar Azam" if "Babar Azam" in names else names[0]
        p = st.selectbox("Player", names, index=names.index(default), key="pi_player")
        x = df[df.player == p]
        s = x.groupby("season").agg(matches=("match_id", "nunique"), **{c: (c, "sum") for c in COMP}, total=("impact_score", "sum")).reset_index()
        s["per_match"] = s.total / s.matches
        ui.kpis([("Matches", f"{len(x)}"), ("Total impact", f"{x.impact_score.sum():.0f}"), ("Per match", f"{x.impact_score.mean():.1f}"),
                 ("Best match", f"{x.impact_score.max():.0f}"), ("POTM awards", f"{int(x.player_of_match.sum())}")])
        long = s.melt(id_vars=["season"], value_vars=COMP, var_name="component", value_name="runs")
        long["component"] = long.component.str.replace("_impact", "").str.replace("_", " ")
        ui.show(px.bar(long, x="season", y="runs", color="component", color_discrete_sequence=list(COMP_COLORS.values()),
                       title=f"{p}: impact by season and component").update_xaxes(type="category"))
        ui.table(x.sort_values("date", ascending=False)[["date", "opponent", "runs", "balls_faced", "wickets", "runs_conceded",
                                                         "catches", "impact_score", "team_win", "player_of_match"]].round(1))

    def method():
        ui.md(f"""
    One row per player per match (every player in the XI, even if they did nothing). **Impact score = batting + bowling + fielding + pressure + win bonus**, all in runs.

    - **Batting** = runs above what an average batter scores from the same balls (same season, innings, over), minus {wv} x (dismissal - expected dismissal). Strike rate and boundaries are already inside this; `strike_rate` and `boundary_pct` are shown but not added again.
    - **Bowling** = runs saved vs the league economy for the same season, innings and over, plus {wv} x (wickets - expected wickets). Dot-ball % and economy vs match average are shown but not added again, for the same reason.
    - **Fielding** = {catch} per catch, {stump} per stumping, {runout} per run-out (shared if two fielders are named). A caught-and-bowled counts as the bowler's catch. Substitute fielders are ignored.
    - **Pressure** = {pressure:.0%} extra weight on batting and bowling impact earned in the **powerplay (overs 1-6) or death (16-20)**, in a **close match** (won by <= 10 runs or <= 3 wickets, or super over) or **knockout**, and by batters in a **successful chase**. Stacks, so a death-over wicket in a close chase counts up to {1 + 3 * pressure:.2f}x. Impact is signed, so failing under pressure costs more too.
    - **Win bonus** = {wb} runs if the team won and the player batted, bowled or took a wicket-related fielding action. Not given for ties or no-results.
    - **Player of the match** is *not* in the score. It is used only to check it: the player of the match has the top score in about half of matches and is in the top 3 in about 86%.

    **Limits**
    - Cricsheet names no fielder on 145 of 342 run-outs. 67 of them are filled from Cricbuzz scorecards (names matched against that team's XI for the match; this method agrees with Cricsheet on 174 of 176 run-outs where both exist). **About 78 run-outs still have no fielder credited.** Dropped catches and misfields are not recorded.
    - `wicketkeeper` is 1/0 for 290 of 358 matches (Cricbuzz) and blank for the rest. It is information only: keepers get credit through catches and stumpings, nothing extra.
    - Non-striker run-outs are not charged to the batter (same as the existing ratings).
    - The weights are judgement calls. Move the sliders and see whether the rankings change.
    - Team-win bonus and pressure partly reward being on a good team in a good match; treat per-match averages over fewer than ~30 matches with care.
    """)

    ui.views({"Leaderboard": leaderboard, "Match explorer": match_view, "Player profile": player_view, "Method": method}, key="pi_view")
    st.download_button("Download player_impact.csv (current weights, selected seasons)",
                       _export(df).encode("utf-8"), "player_impact.csv", "text/csv")
