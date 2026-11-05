"""Analytics + machine-learning layer (scikit-learn only).

Everything that trains a model reports out-of-sample numbers (expanding-window by season,
or grouped K-fold by match) so the dashboard never shows in-sample accuracy as a result.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss, mean_absolute_error,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D

BASE_ELO = 1500.0


# =============================================================================== Elo
def _win_prob(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


@st.cache_data(show_spinner=False)
def elo_table(k: float = 24.0, regress: float = 0.25):
    """Sequential Elo. Returns (per-match table, rating-after-each-match long table)."""
    m = D.load_matches_all()
    m = m[m.decided & m.result_type.isin(["normal", "super over"])].sort_values("date", kind="stable")
    elo, last_season, rows, long = {}, None, [], []
    for r in m.itertuples():
        if last_season is not None and r.season != last_season:
            elo = {t: BASE_ELO + (1 - regress) * (v - BASE_ELO) for t, v in elo.items()}
        last_season = r.season
        ra, rb = elo.get(r.team1, BASE_ELO), elo.get(r.team2, BASE_ELO)
        p1 = _win_prob(ra, rb)
        y = 1.0 if r.winner == r.team1 else 0.0
        elo[r.team1] = ra + k * (y - p1)
        elo[r.team2] = rb + k * ((1 - y) - (1 - p1))
        rows.append(dict(date=r.date, season=r.season, stage=r.stage, team1=r.team1, team2=r.team2,
                         winner=r.winner, elo1=ra, elo2=rb, p1=p1, y=y))
        long += [dict(date=r.date, season=r.season, team=r.team1, elo=elo[r.team1]),
                 dict(date=r.date, season=r.season, team=r.team2, elo=elo[r.team2])]
    return pd.DataFrame(rows), pd.DataFrame(long)


# =============================================================================== match predictor
FEATURES = ["elo_diff", "form_diff", "h2h_diff", "toss_win_t1", "t1_bats_first", "chase_adv"]


def _row_features(S, t1, t2, toss_winner, bat_first, chase_team, venue):
    e1, e2 = S["elo"].get(t1, BASE_ELO), S["elo"].get(t2, BASE_ELO)
    f = lambda t: float(np.mean(S["recent"].get(t, [])[-5:])) if S["recent"].get(t) else 0.5
    a, b = S["h2h"].get((t1, t2), [0, 0]), S["h2h"].get((t2, t1), [0, 0])
    w1, w2 = a[0] + b[1], a[1] + b[0]
    vc, vn = S["venue"].get(venue, [0, 0])
    v_rate = (vc + 3) / (vn + 6)
    ch = 1 if chase_team == t1 else (-1 if chase_team == t2 else 0)
    return {
        "elo_diff": e1 - e2, "form_diff": f(t1) - f(t2),
        "h2h_diff": (w1 + 2) / (w1 + w2 + 4) - 0.5,
        "toss_win_t1": 1.0 if toss_winner == t1 else (0.0 if toss_winner == t2 else 0.5),
        "t1_bats_first": 1.0 if bat_first == t1 else (0.0 if bat_first == t2 else 0.5),
        "chase_adv": (v_rate - 0.5) * ch,
    }


@st.cache_data(show_spinner="Building pre-match features...")
def feature_table(k: float = 24.0, regress: float = 0.25):
    """One row per decided match using ONLY information available before the match."""
    m = D.load_matches_all()
    m = m[m.decided & m.result_type.isin(["normal", "super over"])].sort_values("date", kind="stable")
    S = dict(elo={}, recent={}, h2h={}, venue={})
    last_season, rows = None, []
    for r in m.itertuples():
        if last_season is not None and r.season != last_season:
            S["elo"] = {t: BASE_ELO + (1 - regress) * (v - BASE_ELO) for t, v in S["elo"].items()}
        last_season = r.season
        bat_first = getattr(r, "bat_first", np.nan)
        chase = getattr(r, "chase_team", np.nan)
        feats = _row_features(S, r.team1, r.team2, r.toss_winner, bat_first, chase, r.venue)
        y = int(r.winner == r.team1)
        rows.append({**feats, "season": r.season, "date": r.date, "team1": r.team1, "team2": r.team2,
                     "venue": r.venue, "y": y})
        ra, rb = S["elo"].get(r.team1, BASE_ELO), S["elo"].get(r.team2, BASE_ELO)
        p1 = _win_prob(ra, rb)
        S["elo"][r.team1] = ra + k * (y - p1)
        S["elo"][r.team2] = rb + k * ((1 - y) - (1 - p1))
        for t, won in ((r.team1, y), (r.team2, 1 - y)):
            S["recent"].setdefault(t, []).append(won)
        S["h2h"].setdefault((r.team1, r.team2), [0, 0])
        S["h2h"][(r.team1, r.team2)][0 if y else 1] += 1
        if isinstance(chase, str):
            v = S["venue"].setdefault(r.venue, [0, 0])
            v[0] += int(r.winner == chase); v[1] += 1
    return pd.DataFrame(rows), S


def _models():
    return {
        "Elo only (logistic)": (["elo_diff"], lambda: LogisticRegression()),
        "Logistic (all features)": (FEATURES, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5))),
        "Random forest": (FEATURES, lambda: RandomForestClassifier(n_estimators=300, max_depth=4,
                                                                   min_samples_leaf=6, random_state=0)),
        "Gradient boosting": (FEATURES, lambda: GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
    }


@st.cache_data(show_spinner="Training match-outcome models (expanding window)...")
def predictor_eval(k: float = 24.0, regress: float = 0.25):
    X, S = feature_table(k, regress)
    seasons = sorted(X.season.unique())
    oof, ms = [], _models()
    for s in seasons[2:]:                                  # first two seasons are burn-in
        tr, te = X[X.season < s], X[X.season == s]
        if tr.y.nunique() < 2 or te.empty:
            continue
        part = te[["season", "date", "team1", "team2", "y"]].copy()
        for name, (cols, make) in ms.items():
            mdl = make().fit(tr[cols], tr.y)
            part[name] = mdl.predict_proba(te[cols])[:, 1]
        oof.append(part)
    oof = pd.concat(oof, ignore_index=True)
    rows = [{"model": "Coin flip", "accuracy": 0.5, "log_loss": float(np.log(2)), "brier": 0.25, "auc": 0.5}]
    rows.append({"model": "Always pick higher Elo",
                 "accuracy": float(((oof["Elo only (logistic)"] > 0.5) == oof.y).mean()),
                 "log_loss": np.nan, "brier": np.nan, "auc": np.nan})
    for name in ms:
        p = oof[name].clip(0.02, 0.98)
        rows.append({"model": name, "accuracy": accuracy_score(oof.y, p > 0.5), "log_loss": log_loss(oof.y, p),
                     "brier": brier_score_loss(oof.y, p), "auc": roc_auc_score(oof.y, p)})
    final = {n: make().fit(X[c], X.y) for n, (c, make) in ms.items()}
    rf, lr = final["Random forest"], final["Logistic (all features)"]
    imp = pd.DataFrame({"feature": FEATURES, "rf_importance": rf.feature_importances_,
                        "logit_coef_std": lr[-1].coef_[0]})
    return dict(table=pd.DataFrame(rows), oof=oof, models=final, importance=imp,
                n_train=len(X), n_test=len(oof), state=S)


def calibration(oof: pd.DataFrame, col: str, bins: int = 5) -> pd.DataFrame:
    q = pd.cut(oof[col], np.linspace(0, 1, bins + 1), include_lowest=True)
    return oof.groupby(q, observed=True).agg(pred=(col, "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(drop=True)


def predict_pair(res, model, team1, team2, toss_winner, toss_decision, venue):
    other = team2 if toss_winner == team1 else team1
    bat_first = toss_winner if toss_decision == "bat" else other
    chase = team2 if bat_first == team1 else team1
    f = _row_features(res["state"], team1, team2, toss_winner, bat_first, chase, venue)
    cols, _ = _models()[model]
    return float(res["models"][model].predict_proba(pd.DataFrame([f])[cols])[0, 1]), f


# =============================================================================== in-match models
@st.cache_data(show_spinner="Building ball-by-ball match states...")
def ball_states() -> pd.DataFrame:
    d = D.load_deliveries()
    m = D.load_matches()[["match_idx", "inn1_runs", "winner"]]
    d = d.merge(m, on="match_idx", how="left").sort_values(["match_idx", "innings", "over", "ball"])
    g = d.groupby(["match_idx", "innings"])
    d["cum_runs"] = g.total.cumsum()
    d["cum_wkts"] = g.is_dismissal.cumsum()
    d["cum_balls"] = g.bowl_balls.cumsum()
    d["balls_left"] = (120 - d.cum_balls).clip(lower=0)
    d["wkts_in_hand"] = 10 - d.cum_wkts
    d["crr"] = d.cum_runs * 6 / d.cum_balls.clip(lower=1)
    d["last30"] = g.total.transform(lambda s: s.rolling(30, min_periods=1).sum())
    d["target"] = np.where(d.innings == 2, d.inn1_runs + 1, np.nan)
    d["runs_needed"] = d.target - d.cum_runs
    d["rrr"] = d.runs_needed * 6 / d.balls_left.clip(lower=1)
    d["rr_gap"] = d.rrr - d.crr
    d["y_chase"] = np.where(d.innings == 2, (d.winner == d.bat_team).astype(float), np.nan)
    d.loc[d.winner.isna(), "y_chase"] = np.nan
    vsum = d[d.innings == 1].groupby("match_idx").agg(v=("venue", "first"), r=("inn1_runs", "first"))
    vs = vsum.groupby("v").r.agg(["sum", "count"])
    vsum = vsum.join(vs, on="v")
    loo = ((vsum["sum"] - vsum.r) / (vsum["count"] - 1).clip(lower=1)).rename("venue_avg")
    d = d.merge(loo, left_on="match_idx", right_index=True, how="left")
    return d.reset_index(drop=True)


CH_FEATS = ["runs_needed", "balls_left", "wkts_in_hand", "crr", "rrr", "rr_gap"]
PR_FEATS = ["cum_runs", "cum_wkts", "balls_left", "crr", "last30", "venue_avg"]


@st.cache_data(show_spinner="Training win-probability and score-projection models...")
def inmatch_models():
    s = ball_states()
    # ---- chase win probability
    c = s[(s.innings == 2) & s.y_chase.notna() & (s.runs_needed > 0) & (s.balls_left > 0)].reset_index(drop=True)
    mk = {"Logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)),
          "Gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=3, random_state=0)}
    oof = pd.DataFrame({"match_idx": c.match_idx, "balls_left": c.balls_left, "y": c.y_chase})
    for n, f in mk.items():
        p = np.zeros(len(c))
        for tr, te in GroupKFold(5).split(c, c.y_chase, c.match_idx):
            p[te] = f().fit(c.loc[tr, CH_FEATS], c.y_chase[tr]).predict_proba(c.loc[te, CH_FEATS])[:, 1]
        oof[n] = p
    bucket = pd.cut(oof.balls_left, [0, 30, 60, 90, 120], include_lowest=True)
    ch_rows = [{"model": n, "log_loss": log_loss(oof.y, oof[n].clip(0.01, 0.99)),
                "accuracy": accuracy_score(oof.y, oof[n] > 0.5), "brier": brier_score_loss(oof.y, oof[n])} for n in mk]
    ch_bucket = pd.DataFrame({n: oof.groupby(bucket, observed=True).apply(
        lambda g: accuracy_score(g.y, g[n] > 0.5), include_groups=False) for n in mk})
    ch_bucket.index = ["last 5 overs", "overs 11-15", "overs 6-10", "first 5 overs"]
    chase_models = {n: f().fit(c[CH_FEATS], c.y_chase) for n, f in mk.items()}

    # ---- first-innings score projection
    p1 = s[(s.innings == 1) & (s.cum_balls >= 6)].copy()
    p1["final"] = p1.inn1_runs
    p1 = p1.reset_index(drop=True)
    pred = np.zeros(len(p1))
    for tr, te in GroupKFold(5).split(p1, p1.final, p1.match_idx):
        mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=3, random_state=0)
        pred[te] = mdl.fit(p1.loc[tr, PR_FEATS], p1.final[tr]).predict(p1.loc[te, PR_FEATS])
    p1["pred_ml"] = pred
    p1["pred_crr"] = p1.cum_runs / p1.cum_balls.clip(lower=1) * 120
    p1["over_bucket"] = pd.cut(120 - p1.balls_left, [0, 30, 60, 90, 120], labels=["overs 1-5", "6-10", "11-15", "16-20"], include_lowest=True)
    pr_table = p1.groupby("over_bucket", observed=True).apply(
        lambda g: pd.Series({"ML model MAE": mean_absolute_error(g.final, g.pred_ml),
                             "Run-rate x 20 MAE": mean_absolute_error(g.final, g.pred_crr)}), include_groups=False)
    proj_model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=3, random_state=0)
    proj_model.fit(p1[PR_FEATS], p1.final)
    return dict(chase_models=chase_models, chase_eval=pd.DataFrame(ch_rows), chase_bucket=ch_bucket,
                chase_oof=oof, proj_model=proj_model, proj_eval=pr_table,
                proj_mae=(mean_absolute_error(p1.final, p1.pred_ml), mean_absolute_error(p1.final, p1.pred_crr)))


def chase_prob(res, runs_needed, balls_left, wkts_in_hand, crr, model="Gradient boosting"):
    rrr = runs_needed * 6 / max(balls_left, 1)
    X = pd.DataFrame([dict(runs_needed=runs_needed, balls_left=balls_left, wkts_in_hand=wkts_in_hand,
                           crr=crr, rrr=rrr, rr_gap=rrr - crr)])[CH_FEATS]
    return float(res["chase_models"][model].predict_proba(X)[0, 1])


@st.cache_data(show_spinner="Scoring every chasing delivery...")
def chase_wpa():
    """Win probability after every delivery of every 2nd innings + the change (win probability added)."""
    res = inmatch_models()
    s = ball_states()
    c = s[(s.innings == 2)].copy()
    mdl = res["chase_models"]["Gradient boosting"]
    p = mdl.predict_proba(c[CH_FEATS].fillna(0))[:, 1]
    p = np.where(c.runs_needed <= 0, 1.0, np.where(c.balls_left <= 0, 0.0, p))
    c["wp"] = p
    start = pd.DataFrame({"runs_needed": c.groupby("match_idx").target.first(), "balls_left": 120,
                          "wkts_in_hand": 10, "crr": 0.0, "rrr": c.groupby("match_idx").target.first() * 6 / 120,
                          "rr_gap": c.groupby("match_idx").target.first() * 6 / 120})
    c["wp_prev"] = c.groupby("match_idx").wp.shift(1)
    c["wp_prev"] = c.wp_prev.fillna(c.match_idx.map(pd.Series(mdl.predict_proba(start[CH_FEATS])[:, 1], index=start.index)))
    c["dwp"] = c.wp - c.wp_prev
    return c


# =============================================================================== clustering / similarity
def batter_profiles(d: pd.DataFrame, min_balls: int = 150) -> pd.DataFrame:
    s = D.batting_summary(d)
    s = s[s.balls >= min_balls].copy()
    ph = d.pivot_table(index="batter", columns="phase", values="bat_balls", aggfunc="sum", fill_value=0)
    ph = ph.div(ph.sum(axis=1), axis=0)
    s = s.merge(ph[[D.PHASES[0], D.PHASES[2]]].rename(columns={D.PHASES[0]: "pp_share", D.PHASES[2]: "death_share"}),
                left_on="batter", right_index=True, how="left")
    s["balls_per_inn"] = s.balls / s.innings
    s["six_pct"] = 100 * s.sixes / s.balls
    s["avg"] = s.avg.fillna(s.runs).clip(upper=80)
    s = s.rename(columns={"batter": "player"})
    return s


BAT_FEATS = ["sr", "boundary_pct", "dot_pct", "avg", "balls_per_inn", "six_pct", "pp_share", "death_share"]


def bowler_profiles(d: pd.DataFrame, min_balls: int = 120) -> pd.DataFrame:
    s = D.bowling_summary(d)
    s = s[s.balls >= min_balls].copy()
    ph = d.pivot_table(index="bowler", columns="phase", values="bowl_balls", aggfunc="sum", fill_value=0)
    ph = ph.div(ph.sum(axis=1), axis=0)
    s = s.merge(ph[[D.PHASES[0], D.PHASES[2]]].rename(columns={D.PHASES[0]: "pp_share", D.PHASES[2]: "death_share"}),
                left_on="bowler", right_index=True, how="left")
    s["bowl_sr"] = s.bowl_sr.fillna(s.balls).clip(upper=60)
    return s.rename(columns={"bowler": "player"})


BOWL_FEATS = ["economy", "dot_pct", "boundary_pct", "bowl_sr", "pp_share", "death_share"]


def run_cluster(prof: pd.DataFrame, feats, k: int):
    X = StandardScaler().fit_transform(prof[feats].fillna(prof[feats].median()))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    out = prof.copy()
    out["cluster"] = km.labels_
    out["pc1"], out["pc2"] = xy[:, 0], xy[:, 1]
    z = pd.DataFrame(km.cluster_centers_, columns=feats)
    return out, z


def silhouette_scan(prof, feats, ks=range(2, 8)):
    X = StandardScaler().fit_transform(prof[feats].fillna(prof[feats].median()))
    return pd.DataFrame({"k": list(ks), "silhouette": [
        silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)) for k in ks]})


def label_clusters(z: pd.DataFrame) -> dict:
    names = {}
    for i, row in z.iterrows():
        top = row.abs().sort_values(ascending=False).index[:2]
        names[i] = f"C{i + 1}: " + ", ".join(f"{'high' if row[f] > 0 else 'low'} {f}" for f in top)
    return names


def nearest_players(prof, feats, player, n=8):
    X = StandardScaler().fit_transform(prof[feats].fillna(prof[feats].median()))
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)
    i = prof.index[prof.player == player][0]
    pos = prof.index.get_loc(i)
    sim = X @ X[pos]
    out = prof.assign(similarity=sim).drop(index=i).sort_values("similarity", ascending=False).head(n)
    return out


# =============================================================================== impact (runs above average)
@st.cache_data(show_spinner="Computing impact ratings...")
def impact_tables(seasons: tuple, wicket_value: float = 18.0):
    """Context-adjusted value: runs above the league average for the same season, innings and over.
    Batter: runs above average minus the cost of dismissals above average.
    Bowler: runs saved plus wickets above average (valued at `wicket_value` runs)."""
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)].copy()
    key = ["season", "innings", "over"]
    g = d.groupby(key).agg(rb=("runs_batter", "sum"), bb=("bat_balls", "sum"), br=("bowl_runs", "sum"),
                           bl=("bowl_balls", "sum"), w=("bowler_wkt", "sum"), o=("batter_out", "sum")).reset_index()
    g["e_rb"] = g.rb / g.bb.clip(lower=1)
    g["e_br"] = g.br / g.bl.clip(lower=1)
    g["e_w"] = g.w / g.bl.clip(lower=1)
    g["e_o"] = g.o / g.bb.clip(lower=1)
    d = d.merge(g[key + ["e_rb", "e_br", "e_w", "e_o"]], on=key, how="left")
    d["raa_runs"] = d.runs_batter - d.e_rb * d.bat_balls
    d["exp_out"] = d.e_o * d.bat_balls
    d["saved"] = d.e_br * d.bowl_balls - d.bowl_runs
    d["w_above"] = d.bowler_wkt - d.e_w * d.bowl_balls
    bat = d.groupby("batter").agg(balls=("bat_balls", "sum"), runs=("runs_batter", "sum"), raa_runs=("raa_runs", "sum"),
                                  outs=("batter_out", "sum"), exp_out=("exp_out", "sum"), innings=("match_idx", "nunique")).reset_index()
    bat["bat_impact"] = bat.raa_runs - wicket_value * (bat.outs - bat.exp_out)
    bat["bat_impact_per100"] = 100 * bat.bat_impact / bat.balls.clip(lower=1)
    bowl = d.groupby("bowler").agg(balls=("bowl_balls", "sum"), saved=("saved", "sum"), wkts=("bowler_wkt", "sum"),
                                   w_above=("w_above", "sum"), matches=("match_idx", "nunique")).reset_index()
    bowl["bowl_impact"] = bowl.saved + wicket_value * bowl.w_above
    bowl["bowl_impact_per100"] = 100 * bowl.bowl_impact / bowl.balls.clip(lower=1)
    allp = bat[["batter", "balls", "bat_impact"]].rename(columns={"batter": "player", "balls": "bat_balls"}).merge(
        bowl[["bowler", "balls", "bowl_impact"]].rename(columns={"bowler": "player", "balls": "bowl_balls"}),
        on="player", how="outer").fillna(0)
    allp["total_impact"] = allp.bat_impact + allp.bowl_impact
    return bat, bowl, allp


# =============================================================================== anomalies
def isolation_scores(df: pd.DataFrame, feats, contamination: float = 0.04):
    X = StandardScaler().fit_transform(df[feats].fillna(df[feats].median()))
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=0).fit(X)
    out = df.copy()
    out["anomaly_score"] = -iso.score_samples(X)
    out["is_anomaly"] = iso.predict(X) == -1
    z = pd.DataFrame(X, columns=feats, index=df.index)
    out["why"] = [", ".join(f"{f} {'high' if z.loc[i, f] > 0 else 'low'} ({z.loc[i, f]:+.1f}σ)"
                            for f in z.loc[i].abs().sort_values(ascending=False).index[:2]) for i in df.index]
    return out
