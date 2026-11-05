import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import data as D, ml, ui


def render():
    c1, c2 = st.columns(2)
    k = c1.slider("Elo K-factor", 8, 40, 24, 2)
    reg = c2.slider("Elo carry-over between seasons", 0.0, 0.6, 0.25, 0.05)
    res = ml.predictor_eval(float(k), float(reg))
    tbl = res["table"]
    real = tbl[~tbl.model.isin(["Coin flip", "Always pick higher Elo"])]
    best = real.sort_values("log_loss").iloc[0]
    ui.kpis([("Matches used to train", res["n_train"]), ("Matches predicted out-of-sample", res["n_test"]),
             ("Best accuracy", f"{100 * real.accuracy.max():.1f}%"), ("Best log loss", f"{best.log_loss:.3f}", "0.693 = a coin flip")])
    if best.log_loss >= np.log(2):
        st.warning("**No model beats a coin flip on log loss.** Accuracy a few points above 50% is within the noise for this many matches. "
                   "Pre-match information (team rating, form, toss, venue) carries very little signal in this league. "
                   "Use the probabilities below as a rough guide, not a forecast.")
    else:
        st.info("The best model has a log loss below a coin flip's 0.693, but the improvement is small. Treat it as a weak signal.")

    st.markdown("#### How the models compare")
    st.caption("Method: expanding window. For each season from the third onward, models train on all earlier seasons and predict that season.")
    ui.table(tbl.round(3))

    t1, t2, t3 = st.tabs(["Calibration & accuracy", "What drives it", "Predict a match"])
    with t1:
        model = st.selectbox("Model", list(real.model))
        cal = ml.calibration(res["oof"], model)
        fig = px.scatter(cal, x="pred", y="actual", size="n", title="When the model says X%, how often does the team win?")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
        fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
        ui.show(fig)
        o = res["oof"].assign(correct=lambda x: (x[model] > 0.5) == (x.y == 1)).groupby("season").correct.agg(["mean", "size"]).reset_index()
        ui.show(px.bar(o, x="season", y="mean", text="size", title="Accuracy by season (labels = matches)"))
    with t2:
        imp = res["importance"]
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(imp.sort_values("rf_importance"), x="rf_importance", y="feature", orientation="h", title="Random forest importance"))
        with c2:
            ui.show(px.bar(imp.sort_values("logit_coef_std"), x="logit_coef_std", y="feature", orientation="h", title="Logistic coefficients (standardised)"))
        st.caption("Features: Elo difference, recent form (last 5), head-to-head record, toss winner, who bats first, and the venue's historical chasing success. "
                   "team1 is the first team listed for the match, so a positive value favours team1.")
    with t3:
        m = D.load_matches_all()
        a = st.selectbox("Team 1", D.TEAMS, index=2)
        b = st.selectbox("Team 2", [t for t in D.TEAMS if t != a], index=3)
        tw = st.radio("Toss won by", [a, b], horizontal=True)
        dec = st.radio("Decision", ["field", "bat"], horizontal=True)
        venue = st.selectbox("Venue", sorted(m.venue.dropna().unique()))
        mdl = st.selectbox("Model for prediction", list(real.model), index=1, key="pm")
        p, f = ml.predict_pair(res, mdl, a, b, tw, dec, venue)
        fig = go.Figure(go.Bar(x=[100 * p, 100 * (1 - p)], y=[a, b], orientation="h", marker_color=[D.TEAM_COLORS.get(a, "#1F6B45"), D.TEAM_COLORS.get(b, "#C9962E")],
                               text=[f"{100 * p:.0f}%", f"{100 * (1 - p):.0f}%"], textposition="inside"))
        fig.update_layout(height=220, xaxis=dict(range=[0, 100], title="win probability %"), margin=dict(l=0, r=0, t=10, b=0))
        ui.show(fig)
        ui.table(pd.DataFrame([f]).round(3))
        st.caption("Uses ratings and form as of the last match in the data. Expect probabilities between about 35% and 65%: that is all this data supports.")

