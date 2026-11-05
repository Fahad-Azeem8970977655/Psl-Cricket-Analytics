import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import binomtest
from lib import data as D, ui


def render():
    seasons = ui.selected_seasons()
    m = D.load_matches()
    m = m[m.season.isin(seasons) & m.decided & m.toss_winner.notna()]
    n = len(m)


    def ci(k, n):
        t = binomtest(int(k), int(n), 0.5)
        lo, hi = t.proportion_ci(0.95)
        return t.pvalue, lo, hi


    p1, lo1, hi1 = ci(m.toss_winner_won.sum(), n)
    p2, lo2, hi2 = ci(m.chase_won.sum(), n)
    ui.kpis([("Decided matches", n),
             ("Toss winner won", f"{100 * m.toss_winner_won.mean():.1f}%", f"95% CI {100 * lo1:.0f}-{100 * hi1:.0f}%, p={p1:.2f} vs 50%"),
             ("Chasing team won", f"{100 * m.chase_won.mean():.1f}%", f"95% CI {100 * lo2:.0f}-{100 * hi2:.0f}%, p={p2:.2f} vs 50%"),
             ("Chose to field", f"{100 * (m.toss_decision == 'field').mean():.0f}%")])
    st.caption("A p-value above about 0.05 means the gap from 50% could easily be chance with this many matches.")

    t1, t2, t3 = st.tabs(["Toss & chase", "Head-to-head matrix", "Pair detail"])
    with t1:
        by = m.groupby("season").agg(chase_win=("chase_won", "mean"), field_choice=("toss_decision", lambda s: (s == "field").mean()),
                                     toss_win=("toss_winner_won", "mean"), matches=("chase_won", "size")).reset_index()
        fig = px.line(by.melt(["season", "matches"], var_name="measure", value_name="share"), x="season", y="share", color="measure",
                      markers=True, title="Share of matches by season", hover_data=["matches"])
        fig.add_hline(y=0.5, line_dash="dot")
        ui.show(fig)
        c1, c2 = st.columns(2)
        with c1:
            g = m.groupby("toss_decision").agg(toss_winner_won=("toss_winner_won", "mean"), matches=("chase_won", "size")).reset_index()
            ui.show(px.bar(g, x="toss_decision", y="toss_winner_won", text="matches", title="Toss winner's win rate by decision"))
        with c2:
            g = m.groupby("stage").agg(chase_win=("chase_won", "mean"), matches=("chase_won", "size")).reset_index()
            ui.show(px.bar(g, x="stage", y="chase_win", text="matches", title="Chasing win rate by stage"))

    with t2:
        rows = []
        for a in D.TEAMS:
            for b in D.TEAMS:
                if a == b:
                    continue
                g = m[((m.team1 == a) & (m.team2 == b)) | ((m.team1 == b) & (m.team2 == a))]
                rows.append(dict(team=a, opponent=b, wins=int((g.winner == a).sum()), games=len(g)))
        h = pd.DataFrame(rows)
        h["win_pct"] = np.where(h.games > 0, 100 * h.wins / h.games.replace(0, np.nan), np.nan)
        piv = h.pivot(index="team", columns="opponent", values="win_pct")
        fig = px.imshow(piv.round(0), text_auto=True, aspect="auto", color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                        title="Row team's win % against column team")
        ui.show(fig)
        st.caption("Small samples: most pairs have under 15 meetings. Read the games count in the pair detail tab.")

    with t3:
        a = st.selectbox("Team A", D.TEAMS, index=0)
        b = st.selectbox("Team B", [t for t in D.TEAMS if t != a], index=0)
        g = m[((m.team1 == a) & (m.team2 == b)) | ((m.team1 == b) & (m.team2 == a))].sort_values("date")
        ui.kpis([(f"{a} wins", int((g.winner == a).sum())), (f"{b} wins", int((g.winner == b).sum())), ("Meetings", len(g))])
        ui.table(g[["date", "stage", "venue", "toss_winner", "toss_decision", "winner", "margin_type", "margin_value"]].assign(date=lambda x: x.date.dt.date))

