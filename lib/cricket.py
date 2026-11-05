"""Match-level and player-level analytics built on the exact delivery data."""
import difflib

import numpy as np
import pandas as pd
import streamlit as st

from . import data as D
from . import ml


def overs_text(balls) -> str:
    b = int(round(balls))
    return f"{b // 6}.{b % 6}"


def how_out(r) -> str:
    k, f1, f2 = r.wicket_type, r.fielder_1, r.fielder_2
    fld = f"sub ({f1})" if (r.fielder_is_substitute and isinstance(f1, str)) else (f1 if isinstance(f1, str) else "?")
    if k == "caught":
        return f"c {fld} b {r.bowler}"
    if k == "caught and bowled":
        return f"c & b {r.bowler}"
    if k == "bowled":
        return f"b {r.bowler}"
    if k == "lbw":
        return f"lbw b {r.bowler}"
    if k == "stumped":
        return f"st {fld} b {r.bowler}"
    if k == "hit wicket":
        return f"hit wicket b {r.bowler}"
    if k == "run out":
        names = [n for n in (f1, f2) if isinstance(n, str)]
        return f"run out ({'/'.join(names)})" if names else "run out"
    return str(k)


# =============================================================================== scorecards
@st.cache_data(show_spinner=False)
def scorecard(match_idx: int) -> dict:
    d = D.load_deliveries()
    x = d[d.match_idx == match_idx]
    xi = D.load_xi()
    out = {}
    for inn in (1, 2):
        y = x[x.innings == inn].reset_index(drop=True).copy()
        if y.empty:
            continue
        y["cum_runs"] = y.total.cumsum()
        y["legal_cum"] = y.bowl_balls.cumsum()
        y["legal_in_over"] = y.groupby("over").is_legal.cumsum()
        team, opp = y.bat_team.iloc[0], y.bowl_team.iloc[0]

        seq = pd.concat([y[["batter"]].rename(columns={"batter": "p"}),
                         y[["non_striker"]].rename(columns={"non_striker": "p"})]).sort_index(kind="stable")
        order = seq.p.drop_duplicates().tolist()
        g = y.groupby("batter").agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum"),
                                    fours=("is_four", "sum"), sixes=("is_six", "sum"))
        ev = y[y.wicket_type.notna()]
        rows = []
        for pos, p in enumerate(order, 1):
            e = ev[ev.player_out == p].tail(1)
            r = g.loc[p] if p in g.index else None
            runs, balls = (int(r.runs), int(r.balls)) if r is not None else (0, 0)
            rows.append({"#": pos, "batter": p, "how out": how_out(e.iloc[0]) if len(e) else "not out",
                         "runs": runs, "balls": balls, "4s": int(r.fours) if r is not None else 0,
                         "6s": int(r.sixes) if r is not None else 0,
                         "SR": round(100 * runs / balls, 1) if balls else np.nan})
        bat = pd.DataFrame(rows)

        w = y[y.is_dismissal]
        fow = pd.DataFrame([{"wicket": i + 1, "score": int(r.cum_runs), "batter": r.player_out,
                             "over": f"{int(r.over) - 1}.{int(r.legal_in_over)}"} for i, r in enumerate(w.itertuples())])
        bo = y.groupby("bowler", sort=False).agg(balls=("bowl_balls", "sum"), runs=("bowl_runs", "sum"),
                                                 wkts=("bowler_wkt", "sum"), dots=("is_dot_b", "sum"), wides=("wides", "sum"),
                                                 noballs=("noballs", "sum"), fours=("is_four", "sum"), sixes=("is_six", "sum")).reset_index()
        mo = y.groupby(["over", "bowler"]).agg(b=("bowl_balls", "sum"), r=("bowl_runs", "sum")).reset_index()
        maid = mo[(mo.b >= 6) & (mo.r == 0)].groupby("bowler").size()
        bo["maidens"] = bo.bowler.map(maid).fillna(0).astype(int)
        bo["overs"] = bo.balls.map(overs_text)
        bo["economy"] = (bo.runs / (bo.balls / 6)).round(2)
        bowl = bo[["bowler", "overs", "maidens", "runs", "wkts", "economy", "dots", "wides", "noballs", "fours", "sixes"]]

        ph = y.groupby("phase").agg(runs=("total", "sum"), wkts=("is_dismissal", "sum"), balls=("bowl_balls", "sum")).reindex(D.PHASES).reset_index()
        ph["run_rate"] = (ph.runs * 6 / ph.balls).round(2)
        played = xi[(xi.match_idx == match_idx) & (xi.team == team)].player.tolist()
        total_balls = y.bowl_balls.sum()
        out[inn] = dict(
            team=team, opp=opp, batting=bat, fow=fow, bowling=bowl, phases=ph,
            did_not_bat=[p for p in played if p not in order],
            extras=dict(byes=int(y.byes.sum()), leg_byes=int(y.legbyes.sum()), wides=int(y.wides.sum()),
                        no_balls=int(y.noballs.sum()), penalty=int(y.penalty.sum()), total=int(y.extras.sum())),
            total=dict(runs=int(y.total.sum()), wkts=int(y.is_dismissal.sum()), overs=overs_text(total_balls),
                       run_rate=round(y.total.sum() * 6 / max(total_balls, 1), 2)),
            worm=pd.DataFrame({"ball": y.legal_cum, "runs": y.cum_runs, "wicket": y.is_dismissal, "who": y.player_out}),
            by_over=y.groupby("over").agg(runs=("total", "sum"), wkts=("is_dismissal", "sum")).reset_index(),
        )
    return out


@st.cache_data(show_spinner=False)
def super_over(match_idx: int) -> pd.DataFrame:
    so = D.load_super_over_deliveries()
    so = so[so.match_idx == match_idx]
    return so[["innings", "bat_team", "bowler", "batter", "runs_batter", "extras", "total", "wicket_type"]].reset_index(drop=True)


# =============================================================================== partnerships
@st.cache_data(show_spinner="Building partnerships...")
def partnership_table() -> pd.DataFrame:
    cols = ["match_idx", "season", "date", "innings", "bat_team", "bowl_team", "batter", "non_striker",
            "runs_batter", "total", "bat_balls", "is_dismissal", "player_out", "wicket_type"]
    d = D.load_deliveries()[cols].reset_index(drop=True)
    a, b = d.batter.to_numpy(), d.non_striker.to_numpy()
    d["lo"] = np.where(a < b, a, b)
    d["hi"] = np.where(a < b, b, a)
    d["key"] = d.lo + "|" + d.hi
    d["seg"] = (d.key != d.groupby(["match_idx", "innings"]).key.shift()).cumsum()
    d["wk_before"] = d.groupby(["match_idx", "innings"]).is_dismissal.cumsum() - d.is_dismissal
    g = d.groupby("seg")
    first = g[["match_idx", "season", "date", "innings", "bat_team", "bowl_team", "lo", "hi", "wk_before"]].first()
    agg = g.agg(runs=("total", "sum"), balls=("bat_balls", "sum"), last_dis=("is_dismissal", "last"),
                last_out=("player_out", "last"), last_kind=("wicket_type", "last"))
    t = first.join(agg).reset_index()
    bc = d.groupby(["seg", "batter"]).agg(r=("runs_batter", "sum"), b=("bat_balls", "sum")).reset_index()
    t = t.merge(bc.rename(columns={"batter": "lo", "r": "runs_lo", "b": "balls_lo"}), on=["seg", "lo"], how="left")
    t = t.merge(bc.rename(columns={"batter": "hi", "r": "runs_hi", "b": "balls_hi"}), on=["seg", "hi"], how="left")
    t[["runs_lo", "balls_lo", "runs_hi", "balls_hi"]] = t[["runs_lo", "balls_lo", "runs_hi", "balls_hi"]].fillna(0)
    t["wicket_no"] = t.wk_before + 1
    last_seg = t.groupby(["match_idx", "innings"]).seg.transform("max")
    t["ended"] = np.where(t.last_dis, "wicket", np.where(t.seg == last_seg, "not out / innings end", "retired"))
    t["pair"] = t.lo + " & " + t.hi
    t["extras"] = t.runs - t.runs_lo - t.runs_hi
    t["balls"] = t.balls.round(0)
    return t.rename(columns={"lo": "batter_a", "hi": "batter_b"})


# =============================================================================== batting order
@st.cache_data(show_spinner="Working out batting orders...")
def batting_order_table() -> pd.DataFrame:
    d = D.load_deliveries().reset_index(drop=True)
    d["row"] = np.arange(len(d))
    gk = ["match_idx", "innings"]
    d["runs_before"] = d.groupby(gk).total.cumsum() - d.total
    d["wk_before"] = d.groupby(gk).is_dismissal.cumsum() - d.is_dismissal
    base = ["match_idx", "season", "innings", "bat_team", "row", "over", "runs_before", "wk_before"]
    a = d[base + ["batter"]].rename(columns={"batter": "player"}).assign(k=0)
    b = d[base + ["non_striker"]].rename(columns={"non_striker": "player"}).assign(k=1)
    L = pd.concat([a, b]).sort_values(["match_idx", "innings", "row", "k"], kind="stable")
    f = L.drop_duplicates(["match_idx", "innings", "player"]).copy()
    f["position"] = f.groupby(gk).cumcount() + 1
    perf = d.groupby(gk + ["batter"]).agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum"),
                                          fours=("is_four", "sum"), sixes=("is_six", "sum"), out=("batter_out", "max")).reset_index()
    f = f.merge(perf.rename(columns={"batter": "player"}), on=gk + ["player"], how="left")
    f[["runs", "balls", "fours", "sixes"]] = f[["runs", "balls", "fours", "sixes"]].fillna(0)
    f["out"] = f.out.fillna(False).astype(bool)
    f = f.rename(columns={"over": "entry_over", "runs_before": "entry_runs", "wk_before": "entry_wkts"})
    f["band"] = np.select([f.position <= 2, f.position <= 4, f.position <= 6],
                          ["Openers (1-2)", "Top order (3-4)", "Middle order (5-6)"], "Lower order (7-11)")
    return f.drop(columns=["row", "k"])


BANDS = ["Openers (1-2)", "Top order (3-4)", "Middle order (5-6)", "Lower order (7-11)"]


# =============================================================================== fielding
@st.cache_data(show_spinner="Summarising fielding...")
def fielding_tables(seasons: tuple) -> dict:
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    w = d[d.is_dismissal].copy()
    sub = w.fielder_is_substitute
    xi = D.load_xi()
    xi = xi[xi.season.isin(seasons)]
    matches = xi.groupby("player").match_idx.nunique().rename("matches")
    catches = w[(w.wicket_type == "caught") & w.fielder_1.notna() & ~sub].groupby("fielder_1").size().rename("catches")
    cnb = w[w.wicket_type == "caught and bowled"].groupby("bowler").size().rename("caught_and_bowled")
    stump = w[(w.wicket_type == "stumped") & w.fielder_1.notna() & ~sub].groupby("fielder_1").size().rename("stumpings")
    ro = w[w.wicket_type == "run out"]
    ro1 = ro[ro.fielder_1.notna() & ~ro.fielder_is_substitute].groupby("fielder_1").size()
    ro2 = ro[ro.fielder_2.notna()].groupby("fielder_2").size()
    run_outs = ro1.add(ro2, fill_value=0).rename("run_outs")
    direct = ro[ro.fielder_1.notna() & ro.fielder_2.isna() & ~ro.fielder_is_substitute].groupby("fielder_1").size().rename("direct_hits")
    p = pd.concat([catches, cnb, stump, run_outs, direct], axis=1).fillna(0).astype(int)
    p.index.name = "player"
    p = p.join(matches, how="left").reset_index()
    p["matches"] = p.matches.fillna(0).astype(int)
    p["dismissals"] = p.catches + p.caught_and_bowled + p.stumpings + p.run_outs
    p["per_match"] = (p.dismissals / p.matches.replace(0, np.nan)).round(2)
    team_matches = xi.groupby("team").match_idx.nunique().rename("matches_fielded")
    t = w.groupby("bowl_team").agg(
        catches=("wicket_type", lambda s: (s == "caught").sum()), caught_and_bowled=("wicket_type", lambda s: (s == "caught and bowled").sum()),
        stumpings=("wicket_type", lambda s: (s == "stumped").sum()), run_outs=("wicket_type", lambda s: (s == "run out").sum()),
        bowled_lbw=("wicket_type", lambda s: s.isin(["bowled", "lbw"]).sum())).join(team_matches).reset_index().rename(columns={"bowl_team": "team"})
    t["fielding_dismissals_per_match"] = ((t.catches + t.caught_and_bowled + t.stumpings + t.run_outs) / t.matches_fielded).round(2)
    combos = w[(w.wicket_type == "caught") & w.fielder_1.notna() & ~sub].groupby(["bowler", "fielder_1"]).size().rename("catches").reset_index()
    mix = w.groupby(["season", "wicket_type"]).size().rename("wickets").reset_index()
    by_over = w[w.wicket_type.isin(["caught", "run out", "stumped"])].groupby(["wicket_type", "over"]).size().rename("wickets").reset_index()
    return dict(players=p, teams=t, combos=combos, mix=mix, by_over=by_over,
                sub_wickets=int(sub.sum()), run_outs_total=len(ro), run_outs_no_fielder=int(ro.fielder_1.isna().sum()))


# =============================================================================== luck vs skill
@st.cache_data(show_spinner="Simulating coin-flip leagues...")
def team_luck(n_sims: int = 5000, seed: int = 7):
    m = D.load_matches_all()
    lg = m[(m.stage == "league") & m.decided & m.result_type.isin(["normal", "super over"])]
    rng = np.random.default_rng(seed)
    rows, sims = [], {}
    for s, g in lg.groupby("season"):
        teams = sorted(set(g.team1) | set(g.team2))
        idx = {t: i for i, t in enumerate(teams)}
        t1, t2 = g.team1.map(idx).to_numpy(), g.team2.map(idx).to_numpy()
        win1 = (g.winner == g.team1).to_numpy()
        M, T = len(g), len(teams)
        A1, A2 = np.zeros((M, T)), np.zeros((M, T))
        A1[np.arange(M), t1] = 1
        A2[np.arange(M), t2] = 1
        wins = win1.astype(float) @ A1 + (~win1).astype(float) @ A2
        games = A1.sum(0) + A2.sum(0)
        bits = (rng.random((n_sims, M)) < 0.5).astype(float)
        sim_wins = bits @ A1 + (1 - bits) @ A2
        sd_obs, sd_sim = wins.std(), sim_wins.std(axis=1)
        var_obs, var_luck = (wins / games).var(), (0.25 / games).mean()
        sims[s] = sd_sim
        rows.append(dict(season=int(s), teams=T, games_per_team=round(games.mean(), 1), sd_wins_observed=round(sd_obs, 2),
                         sd_wins_if_coin_flips=round(sd_sim.mean(), 2), p_value=round(float((sd_sim >= sd_obs).mean()), 3),
                         var_obs=var_obs, var_luck=var_luck, luck_share=min(var_luck / var_obs, 1.0) if var_obs > 0 else 1.0,
                         true_talent_sd=float(np.sqrt(max(var_obs - var_luck, 0)))))
    df = pd.DataFrame(rows)
    pooled_obs = df.sd_wins_observed.mean()
    pooled_sim = np.mean([sims[s] for s in sims], axis=0)
    pooled = dict(luck_share=float(df.var_luck.mean() / df.var_obs.mean()), true_talent_sd=float(np.sqrt(max(df.var_obs.mean() - df.var_luck.mean(), 0))),
                  observed_sd=float(pooled_obs), sim_sd_mean=float(pooled_sim.mean()), p_value=float((pooled_sim >= pooled_obs).mean()))
    return df, pooled, pooled_sim


def team_persistence() -> dict:
    pt = D.load_points().copy()
    pt["win_pct"] = pt.won / (pt.played - pt.no_result).replace(0, np.nan)
    nxt = pt[["season", "team", "win_pct"]].assign(season=lambda x: x.season - 1).rename(columns={"win_pct": "next"})
    j = pt.merge(nxt, on=["season", "team"]).dropna(subset=["win_pct", "next"])
    r = float(np.corrcoef(j.win_pct, j.next)[0, 1])
    slope = float(np.polyfit(j.win_pct, j.next, 1)[0])
    return dict(r=r, slope=slope, n=len(j), data=j)


PLAYOFF_TITLE_ODDS = {1: 0.375, 2: 0.375, 3: 0.125, 4: 0.125}


def playoff_luck() -> pd.DataFrame:
    fr = D.load_final_results()
    pt = D.load_points()
    c = fr[["season", "champion"]].merge(pt, left_on=["season", "champion"], right_on=["season", "team"], how="left")
    cnt = c.final_position.value_counts().reindex([1, 2, 3, 4], fill_value=0)
    n = int(cnt.sum())
    return pd.DataFrame({"league_position": cnt.index, "titles_won": cnt.values,
                         "expected_if_playoffs_were_coin_flips": [round(PLAYOFF_TITLE_ODDS[i] * n, 1) for i in cnt.index]})


@st.cache_data(show_spinner=False)
def _batting_halves(seasons: tuple):
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    pm = d.groupby(["batter", "match_idx"]).agg(date=("date", "first"), runs=("runs_batter", "sum"), balls=("bat_balls", "sum"),
                                               b=("is_four", "sum"), s=("is_six", "sum"), dots=("is_dot", "sum")).reset_index()
    pm = pm.sort_values(["batter", "date", "match_idx"])
    pm["half"] = pm.groupby("batter").cumcount() % 2
    pm["n"] = 1
    return pm.groupby(["batter", "half"])[["runs", "balls", "b", "s", "dots", "n"]].sum().unstack("half")


def _reliability(half: pd.DataFrame, metrics: dict, ball_col: str, min_half: int):
    rows = []
    ok = (half[(ball_col, 0)] >= min_half) & (half[(ball_col, 1)] >= min_half)
    h = half[ok]
    for name, fn in metrics.items():
        a, b = fn(h, 0), fn(h, 1)
        v = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 15:
            continue
        r = float(np.corrcoef(v.iloc[:, 0], v.iloc[:, 1])[0, 1])
        rel = 2 * r / (1 + r) if r > -1 else np.nan
        avg_total = float((h[(ball_col, 0)] + h[(ball_col, 1)]).loc[v.index].mean())
        rows.append(dict(metric=name, players=len(v), split_half_r=r, full_reliability=rel, avg_balls=avg_total,
                         balls_for_50pct_reliability=(avg_total * (1 - rel) / rel) if rel and rel > 0 else np.nan))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Measuring how reliable batting stats are...")
def batting_reliability(seasons: tuple, min_half: int = 40) -> pd.DataFrame:
    h = _batting_halves(seasons)
    mt = {
        "Strike rate": lambda h, k: 100 * h[("runs", k)] / h[("balls", k)],
        "Boundary %": lambda h, k: 100 * (h[("b", k)] + h[("s", k)]) / h[("balls", k)],
        "Six %": lambda h, k: 100 * h[("s", k)] / h[("balls", k)],
        "Dot-ball %": lambda h, k: 100 * h[("dots", k)] / h[("balls", k)],
        "Runs per innings": lambda h, k: h[("runs", k)] / h[("n", k)],
    }
    return _reliability(h, mt, "balls", min_half)


@st.cache_data(show_spinner="Measuring how reliable bowling stats are...")
def bowling_reliability(seasons: tuple, min_half: int = 30) -> pd.DataFrame:
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    pm = d.groupby(["bowler", "match_idx"]).agg(date=("date", "first"), runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum"),
                                               w=("bowler_wkt", "sum"), dots=("is_dot_b", "sum"), b=("is_four", "sum"), s=("is_six", "sum")).reset_index()
    pm = pm.sort_values(["bowler", "date", "match_idx"])
    pm["half"] = pm.groupby("bowler").cumcount() % 2
    h = pm.groupby(["bowler", "half"])[["runs", "balls", "w", "dots", "b", "s"]].sum().unstack("half")
    mt = {
        "Economy": lambda h, k: 6 * h[("runs", k)] / h[("balls", k)],
        "Dot-ball %": lambda h, k: 100 * h[("dots", k)] / h[("balls", k)],
        "Boundary % conceded": lambda h, k: 100 * (h[("b", k)] + h[("s", k)]) / h[("balls", k)],
        "Wickets per 100 balls": lambda h, k: 100 * h[("w", k)] / h[("balls", k)],
    }
    return _reliability(h, mt, "balls", min_half)


@st.cache_data(show_spinner=False)
def year_to_year(seasons: tuple, kind: str = "batting", min_balls: int = 100) -> pd.DataFrame:
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    if kind == "batting":
        g = d.groupby(["batter", "season"]).agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum"), b=("is_four", "sum"),
                                               s=("is_six", "sum"), dots=("is_dot", "sum")).reset_index().rename(columns={"batter": "player"})
        g = g[g.balls >= min_balls]
        g["Strike rate"] = 100 * g.runs / g.balls
        g["Boundary %"] = 100 * (g.b + g.s) / g.balls
        g["Dot-ball %"] = 100 * g.dots / g.balls
        mets = ["Strike rate", "Boundary %", "Dot-ball %"]
    else:
        g = d.groupby(["bowler", "season"]).agg(runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum"), w=("bowler_wkt", "sum"),
                                               dots=("is_dot_b", "sum")).reset_index().rename(columns={"bowler": "player"})
        g = g[g.balls >= min_balls]
        g["Economy"] = 6 * g.runs / g.balls
        g["Dot-ball %"] = 100 * g.dots / g.balls
        g["Wickets per 100 balls"] = 100 * g.w / g.balls
        mets = ["Economy", "Dot-ball %", "Wickets per 100 balls"]
    nxt = g.assign(season=g.season - 1)
    j = g.merge(nxt, on=["player", "season"], suffixes=("", "_next"))
    rows = [dict(metric=m, pairs=len(j), year_to_year_r=float(np.corrcoef(j[m], j[m + "_next"])[0, 1])) for m in mets if len(j) >= 15]
    return pd.DataFrame(rows)


def balls_sd_table(seasons: tuple) -> tuple:
    d = D.load_deliveries()
    d = d[d.season.isin(seasons) & (d.bat_balls > 0)]
    sd = float(d.runs_batter.std())
    sr = 100 * d.runs_batter.sum() / d.bat_balls.sum()
    ns = np.array([30, 60, 100, 200, 400, 800, 1500])
    return sr, sd, pd.DataFrame({"balls faced": ns, "typical error (+/- strike-rate points)": np.round(1.96 * 100 * sd / np.sqrt(ns), 1)})


# =============================================================================== value for money (2026)
@st.cache_data(show_spinner="Matching 2026 prices to performance...")
def value_for_money(wicket_value: float = 18.0, min_matches: int = 3) -> dict:
    a = D.load_auction()
    a = a[(a.season == 2026) & a.final_price_pkr_cr.notna()].copy()
    bat, bowl, allp = ml.impact_tables((2026,), float(wicket_value))
    xi = D.load_xi()
    x26 = xi[xi.season == 2026].groupby("player").agg(matches=("match_idx", "nunique")).reset_index().rename(columns={"player": "canonical"})
    d = D.load_deliveries()
    d26 = d[d.season == 2026]
    bs = D.batting_summary(d26)[["batter", "runs", "balls", "sr"]].rename(columns={"batter": "canonical", "balls": "bat_balls_26"})
    bw = D.bowling_summary(d26)[["bowler", "wkts", "economy", "balls"]].rename(columns={"bowler": "canonical", "balls": "bowl_balls_26"})
    imp = allp.rename(columns={"player": "canonical"})[["canonical", "bat_impact", "bowl_impact", "total_impact"]]
    p = a[["player", "team", "price_type", "base_price_pkr_cr", "final_price_pkr_cr", "canonical", "match_method"]].rename(
        columns={"base_price_pkr_cr": "base_price", "final_price_pkr_cr": "price"})
    for t in (x26, bs, bw, imp):
        p = p.merge(t, on="canonical", how="left")
    p["matched"] = p.canonical.notna()
    p["matches"] = p.matches.fillna(0).astype(int)
    for c in ("bat_impact", "bowl_impact", "total_impact", "runs", "wkts"):
        p[c] = p[c].fillna(0)
    p["status"] = np.select([~p.matched, p.matches == 0, p.matches < min_matches], ["name not matched", "did not play", "played little"], "played")
    use = p[(p.status == "played")]
    fit = None
    p["value_vs_price"] = np.nan
    if len(use) >= 8:
        slope, icpt = np.polyfit(use.price, use.total_impact, 1)
        p.loc[use.index, "value_vs_price"] = use.total_impact - (slope * use.price + icpt)
        fit = dict(slope=float(slope), intercept=float(icpt), pearson=float(use[["price", "total_impact"]].corr().iloc[0, 1]),
                   spearman=float(use[["price", "total_impact"]].corr(method="spearman").iloc[0, 1]), n=len(use))
    p["impact_per_crore"] = np.where(p.status == "played", p.total_impact / p.price, np.nan)

    mt = p[p.matched]
    team = mt.groupby("team").apply(lambda g: pd.Series({
        "players": len(g), "spend_crore": g.price.sum(), "spend_on_players_who_played": g[g.status == "played"].price.sum(),
        "spend_on_bench": g[g.status.isin(["did not play", "played little"])].price.sum(),
        "total_impact": g.total_impact.sum()}), include_groups=False).reset_index()
    team["bench_share_pct"] = (100 * team.spend_on_bench / team.spend_crore).round(1)
    team["impact_per_crore"] = (team.total_impact / team.spend_crore).round(2)

    um = p[~p.matched].copy()
    cand = sorted(set(x26.canonical) - set(p.canonical.dropna()))
    um["possible_match_(unconfirmed)"] = [next(iter(difflib.get_close_matches(n, cand, n=1, cutoff=0.8)), "") for n in um.player]
    cov = dict(rows=len(p), matched_rows=int(p.matched.sum()), crore=float(p.price.sum()), matched_crore=float(p[p.matched].price.sum()))
    return dict(players=p, team=team, fit=fit, coverage=cov, unmatched=um)
