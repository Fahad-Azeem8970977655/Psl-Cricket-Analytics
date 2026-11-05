import numpy as np
import plotly.express as px
import streamlit as st
from lib import data as D, ui


def bubble(x, name_col, title, prior, n_labels=8):
    """Bubble chart with names only on the few points worth reading; everything else is on hover."""
    x = x.copy()
    x["label"] = ""
    keep = x.sort_values(["outs", "balls"], ascending=False).head(n_labels // 2).index.union(
        x.sort_values("balls", ascending=False).head(n_labels - n_labels // 2).index)
    x.loc[keep, "label"] = x.loc[keep, name_col]
    fig = px.scatter(x, x="balls", y="sr_adj", size=(x.outs + 1), color="outs", hover_name=name_col, text="label",
                     hover_data={"runs": True, "sr": ":.0f", "dot_pct": ":.0f", "outs": True, "label": False},
                     color_continuous_scale="Reds", size_max=22, title=title,
                     labels={"balls": "balls faced", "sr_adj": "adjusted strike rate", "outs": "dismissals", "runs": "runs",
                             "sr": "raw strike rate", "dot_pct": "dot %"})
    fig.update_traces(textposition="top center", textfont_size=11, cliponaxis=False)
    fig.add_hline(y=prior, line_dash="dot", annotation_text="overall rate", annotation_position="top left")
    return fig


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)].copy()
    d["out_by_bowler"] = d.bowler_wkt & (d.player_out == d.batter)
    pair = d.groupby(["batter", "bowler"]).agg(balls=("bat_balls", "sum"), runs=("runs_batter", "sum"), dots=("is_dot", "sum"),
                                               fours=("is_four", "sum"), sixes=("is_six", "sum"), outs=("out_by_bowler", "sum")).reset_index()
    pair["balls"] = pair.balls.round(0)
    pair = pair[pair.balls > 0]
    pair["sr"] = 100 * pair.runs / pair.balls
    pair["dot_pct"] = 100 * pair.dots / pair.balls
    pair["balls_per_out"] = pair.balls / pair.outs.replace(0, np.nan)
    league_sr = 100 * pair.runs.sum() / pair.balls.sum()

    mode = st.radio("Mode", ["One batter vs every bowler", "One bowler vs every batter", "League-wide leaderboard"], horizontal=True)
    prior_balls = st.slider("Shrinkage strength (balls of prior)", 0, 60, 20, 5,
                            help="0 = raw numbers. Higher values pull small samples toward the batter's / league's overall strike rate.")


    def shrink(df, prior_sr):
        df = df.copy()
        df["sr_adj"] = (df.runs + prior_sr * prior_balls / 100) / (df.balls + prior_balls) * 100
        return df


    if mode == "One batter vs every bowler":
        bats = pair.groupby("batter").balls.sum().sort_values(ascending=False)
        who = st.selectbox("Batter", bats[bats >= 60].index.tolist())
        min_b = st.slider("Minimum balls faced against a bowler", 1, 40, 12)
        prior = 100 * pair[pair.batter == who].runs.sum() / pair[pair.batter == who].balls.sum()
        x = shrink(pair[(pair.batter == who) & (pair.balls >= min_b)], prior)
        ui.kpis([("Overall strike rate", f"{prior:.1f}"), ("Bowlers shown", len(x)), ("Dismissed by these", int(x.outs.sum()))])
        if len(x):
            fig = bubble(x, "bowler", f"{who} against each bowler (bubble size and colour = dismissals; hover for names)", prior)
            ui.show(fig)
            ui.table(x.sort_values("balls", ascending=False)[["bowler", "balls", "runs", "sr", "sr_adj", "dot_pct", "outs", "fours", "sixes"]].round(1))
        else:
            st.info("No bowler meets the minimum ball count.")
    elif mode == "One bowler vs every batter":
        bw = pair.groupby("bowler").balls.sum().sort_values(ascending=False)
        who = st.selectbox("Bowler", bw[bw >= 60].index.tolist())
        min_b = st.slider("Minimum balls bowled to a batter", 1, 40, 12)
        prior = 100 * pair[pair.bowler == who].runs.sum() / pair[pair.bowler == who].balls.sum()
        x = shrink(pair[(pair.bowler == who) & (pair.balls >= min_b)], prior)
        ui.kpis([("Strike rate conceded", f"{prior:.1f}"), ("Batters shown", len(x)), ("Dismissals", int(x.outs.sum()))])
        if len(x):
            fig = bubble(x, "batter", f"{who} against each batter (bubble size and colour = dismissals; hover for names)", prior)
            ui.show(fig)
            ui.table(x.sort_values("balls", ascending=False)[["batter", "balls", "runs", "sr", "sr_adj", "dot_pct", "outs"]].round(1))
    else:
        min_b = st.slider("Minimum balls in the matchup", 12, 80, 30)
        x = shrink(pair[pair.balls >= min_b], league_sr)
        x["out_rate_per_100"] = 100 * x.outs / x.balls
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Bowler on top** (lowest adjusted strike rate)")
            ui.table(x.sort_values(["sr_adj", "balls"], ascending=[True, False]).head(15)[["batter", "bowler", "balls", "runs", "sr_adj", "outs"]].round(1))
        with c2:
            st.markdown("**Batter on top** (highest adjusted strike rate)")
            ui.table(x.sort_values("sr_adj", ascending=False).head(15)[["batter", "bowler", "balls", "runs", "sr_adj", "outs"]].round(1))
        st.markdown("**Most dismissals of one batter by one bowler**")
        ui.table(x.sort_values(["outs", "balls"], ascending=[False, True]).head(15)[["bowler", "batter", "outs", "balls", "runs"]])
    st.caption("A matchup record is a description, not a prediction: these samples are usually under 50 balls.")

