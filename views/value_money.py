import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import cricket as C, data as D, ui


def render():
    c1, c2, c3 = st.columns(3)
    wv = c1.slider("Value of a wicket (runs)", 8, 30, 18, key="vfm_wv")
    mn = c2.slider("Minimum matches to count as 'played'", 1, 8, 3, key="vfm_mn")
    kind = c3.radio("Players", ["All", "Auction buys", "Retentions"], horizontal=True, key="vfm_kind")
    v = C.value_for_money(float(wv), int(mn))
    p, cov = v["players"], v["coverage"]
    if kind != "All":
        p = p[p.price_type == ("auction" if kind == "Auction buys" else "retention")]

    ui.kpis([("Priced 2026 players", cov["rows"]), ("Matched to a Cricsheet player", f"{cov['matched_rows']} ({100 * cov['matched_rows'] / cov['rows']:.0f}%)"),
             ("Total spend (crore)", f"{cov['crore']:.0f}"), ("Spend we can match (crore)", f"{cov['matched_crore']:.0f} ({100 * cov['matched_crore'] / cov['crore']:.0f}%)")])
    st.warning("2026 prices were matched to a single source (pcb.com.pk). Coaches and owners are mostly confirmed by two sources, but four captains were not re-checked. "
               "Only players whose auction name was matched to a Cricsheet name are scored; near-matches that still need a human check are shown but not used.")
    t1, t2, t3, t4 = st.tabs(["Price vs performance", "Best and worst value", "Teams", "Not matched & 2026 checks"])

    with t1:
        q = p[p.status == "played"]
        if q.empty:
            st.info("No matched players meet the minimum matches.")
        else:
            fig = px.scatter(q, x="price", y="total_impact", color="team", color_discrete_map=D.TEAM_COLORS, hover_name="player",
                             hover_data=["matches", "price_type"], size=q.matches.clip(lower=1), title="2026: price paid vs impact (batting + bowling)")
            if v["fit"]:
                xs = np.array([q.price.min(), q.price.max()])
                fig.add_trace(go.Scatter(x=xs, y=v["fit"]["slope"] * xs + v["fit"]["intercept"], mode="lines", name="trend", line=dict(color="grey", dash="dash")))
            ui.show(fig)
            if v["fit"]:
                f = v["fit"]
                st.caption(f"{f['n']} players: correlation between price and impact is {f['pearson']:.2f} (Spearman {f['spearman']:.2f}). "
                           "Near zero means 2026 price did not track on-field value; with one short season this is a description, not a verdict on how players were priced.")
        st.caption("Impact is runs above average for the same season, innings and over (see Player impact ratings). It ignores batting position and role, and a season is only 10-14 matches.")

    with t2:
        q = p[p.status == "played"].dropna(subset=["value_vs_price"])
        cols = ["player", "team", "price_type", "price", "matches", "runs", "wkts", "total_impact", "value_vs_price"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Delivered most above what their price predicts**")
            ui.table(q.sort_values("value_vs_price", ascending=False).head(10)[cols].round(1))
        with c2:
            st.markdown("**Delivered least for their price**")
            ui.table(q.sort_values("value_vs_price").head(10)[cols].round(1))
        bench = p[p.status.isin(["did not play", "played little"])].sort_values("price", ascending=False)
        st.markdown("**Expensive players who barely played**")
        ui.table(bench[["player", "team", "price_type", "price", "matches", "status"]].head(15))

    with t3:
        tm = v["team"]
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(tm.sort_values("spend_crore"), x="spend_crore", y="team", orientation="h", color="team", color_discrete_map=D.TEAM_COLORS, title="Matched 2026 spend by team (crore)"))
        with c2:
            ui.show(px.bar(tm.sort_values("bench_share_pct"), x="bench_share_pct", y="team", orientation="h", color="team", color_discrete_map=D.TEAM_COLORS,
                           title="Share of spend on players who barely played (%)"))
        ui.table(tm.round(1))
        st.caption("Team totals cover only matched players, so they understate each team's real spend.")

    with t4:
        um = v["unmatched"]
        st.markdown(f"**{len(um)} priced players with no matched Cricsheet name** ({100 * (1 - cov['matched_crore'] / cov['crore']):.0f}% of the money).")
        ui.table(um[["player", "team", "price_type", "price", "possible_match_(unconfirmed)"]].sort_values("price", ascending=False))
        st.caption("'Possible match' is a spelling-similarity guess among players who appeared in 2026. It can be a different person with a similar name, so it is not used until confirmed.")
        ver = D.load_verification()
        st.markdown("**2026 verification file**")
        ui.table(ver.groupby(["item_type", "status"]).size().rename("items").reset_index())
        nf = ver[ver.status != "match"]
        if len(nf):
            ui.table(nf[[c for c in ["item_type", "team", "player_or_person", "my_value", "source_value", "status"] if c in nf.columns]])
