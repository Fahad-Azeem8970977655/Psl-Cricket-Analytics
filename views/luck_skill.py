import pandas as pd
import plotly.express as px
import streamlit as st

from lib import cricket as C, ml, ui


def render():
    seasons = tuple(ui.selected_seasons())
    t1, t2, t3 = st.tabs(["Team results: luck or skill?", "Player stats: how reliable?", "What it means"])
    luck, pooled, sims = C.team_luck()
    pers = C.team_persistence()
    bat_rel = C.batting_reliability(seasons)
    bowl_rel = C.bowling_reliability(seasons)

    with t1:
        ui.kpis([("Share of the win-rate spread explained by luck", f"{100 * pooled['luck_share']:.0f}%",
                  "Variance a league of coin flips would show / variance actually seen"),
                 ("Spread in true team talent", f"{100 * pooled['true_talent_sd']:.1f} win-% points", "Standard deviation of real quality, after removing luck"),
                 ("Chance a pure-luck league looks this uneven", f"{100 * pooled['p_value']:.0f}%", "Share of 5,000 simulated coin-flip leagues with at least this much spread"),
                 ("Last season's win % vs next season's", f"r = {pers['r']:.2f}", f"{pers['n']} team-seasons; 0 = no carry-over")])
        fig = px.histogram(x=sims, nbins=40, title="Average spread of wins if every league match were a coin flip (5,000 simulations)",
                           labels={"x": "average standard deviation of team wins"})
        fig.add_vline(x=pooled["observed_sd"], line_color="#A8242B", annotation_text="what actually happened")
        ui.show(fig)
        ui.table(luck.drop(columns=["var_obs", "var_luck"]).assign(luck_share=lambda x: (100 * x.luck_share).round(0), true_talent_sd=lambda x: (100 * x.true_talent_sd).round(1))
                 .rename(columns={"luck_share": "luck share %", "true_talent_sd": "true talent sd (win-% pts)", "p_value": "p (coin-flip league this uneven)"}))
        st.caption("Each season is a small sample (about 8-10 games per team), so single-season rows are noisy; the pooled numbers above are the reliable summary. "
                   "A p-value near 1 means the league was no more uneven than coin flips would produce.")
        c1, c2 = st.columns(2)
        with c1:
            j = pers["data"]
            ui.show(px.scatter(j, x="win_pct", y="next", hover_data=["team", "season"], trendline=None,
                               title="Win % this season vs next season (same team)", labels={"win_pct": "win % this season", "next": "win % next season"}))
        with c2:
            po = C.playoff_luck()
            ui.show(px.bar(po.melt("league_position", var_name="what", value_name="titles"), x="league_position", y="titles", color="what", barmode="group",
                           title="Titles won by league-table position", color_discrete_sequence=["#1F6B45", "#C9962E"]))
        st.caption("Playoff expectation assumes the usual top-4 page playoff with every match a coin flip (1st and 2nd 37.5% each, 3rd and 4th 12.5% each).")

    with t2:
        ui.md("""
        **Split-half reliability**: each player's matches are alternated into two halves and the same stat is computed in each half.
        If a stat measures real skill, the two halves agree. **Balls for 50% reliability** is the sample at which half of the spread between players is signal and half is noise.
        """)
        c1, c2 = st.columns(2)
        with c1:
            if len(bat_rel):
                ui.show(px.bar(bat_rel, x="metric", y="full_reliability", text=bat_rel.full_reliability.round(2), title="Batting: reliability of a typical sample", range_y=[0, 1]))
                ui.table(bat_rel.round(2))
        with c2:
            if len(bowl_rel):
                ui.show(px.bar(bowl_rel, x="metric", y="full_reliability", text=bowl_rel.full_reliability.round(2), title="Bowling: reliability of a typical sample", range_y=[0, 1]))
                ui.table(bowl_rel.round(2))
        y1, y2 = C.year_to_year(seasons, "batting"), C.year_to_year(seasons, "bowling")
        yy = pd.concat([y1.assign(side="batting"), y2.assign(side="bowling")])
        if len(yy):
            ui.show(px.bar(yy, x="metric", y="year_to_year_r", color="side", barmode="group", title="Correlation of a stat in one season with the same stat the next season (100+ balls each)"))
        sr, sd, tb = C.balls_sd_table(seasons)
        st.markdown(f"**How much noise is in a strike rate?** League strike rate is {sr:.0f}. A batter's true strike rate could easily differ from the one observed by about:")
        ui.table(tb)

    with t3:
        res = ml.predictor_eval(24.0, 0.25)
        real = res["table"][~res["table"].model.isin(["Coin flip", "Always pick higher Elo"])]
        best = real.sort_values("log_loss").iloc[0]
        notes = [
            f"- About **{100 * pooled['luck_share']:.0f}%** of the spread in league win rates is what coin flips would produce; the leftover real talent spread is about **{100 * pooled['true_talent_sd']:.1f} win-% points**.",
            f"- A team's win rate one season says almost nothing about the next (r = {pers['r']:.2f}, {pers['n']} team-seasons), so season-to-season form should not be extrapolated.",
            f"- The best pre-match model reaches {100 * best.accuracy:.0f}% accuracy with log loss {best.log_loss:.3f} (a coin flip is 0.693): consistent with the luck result above.",
        ]
        if len(bat_rel):
            s = bat_rel.sort_values("balls_for_50pct_reliability").iloc[0]
            notes.append(f"- The most stable batting stat here is **{s.metric.lower()}**: it needs about {s.balls_for_50pct_reliability:.0f} balls to be half signal. Anything built on fewer balls is mostly noise.")
        if len(bowl_rel):
            w = bowl_rel.sort_values("balls_for_50pct_reliability").iloc[-1]
            notes.append(f"- The least stable bowling stat is **{w.metric.lower()}** (about {w.balls_for_50pct_reliability:,.0f} balls to be half signal): a bowler's wicket tally in one season is largely luck.")
        notes.append("- Use the impact ratings, clusters and matchup pages as descriptions of what happened, not as forecasts of what will happen.")
        st.markdown("\n".join(notes))
