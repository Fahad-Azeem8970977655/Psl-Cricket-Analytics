"""Match-level player impact score (batting + bowling + fielding + pressure + win bonus).

One row per player per match.  Everything is measured in "runs above an average player", so the
components can simply be added.  The baseline is the league average for the same season, innings
and over (the same idea as the existing run-based ratings on this page).

No Streamlit imports here, so it can be used from a script (build_player_impact.py) as well.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

HIGH_LEVERAGE_OVERS = (6, 16)   # powerplay: over <= 6, death: over >= 16
CLOSE_RUNS = 10                 # won by 10 runs or fewer
CLOSE_WKTS = 3                  # or by 3 wickets or fewer


@dataclass(frozen=True)
class Weights:
    wicket_value: float = 18.0    # runs a wicket / dismissal is worth
    catch: float = 5.0            # runs credited for a catch
    stumping: float = 8.0
    run_out: float = 10.0         # shared equally between the fielders named on the run-out
    pressure: float = 0.25        # extra weight on impact earned under pressure (see below)
    win_bonus: float = 3.0        # runs credited to a contributor on the winning side


def _ball_context(d: pd.DataFrame, w: Weights) -> pd.DataFrame:
    """Add expected-value columns to every delivery (league baseline per season/innings/over)."""
    key = ["season", "innings", "over"]
    g = d.groupby(key).agg(rb=("runs_batter", "sum"), bb=("bat_balls", "sum"), br=("bowl_runs", "sum"),
                           bl=("bowl_balls", "sum"), wk=("bowler_wkt", "sum"), o=("batter_out", "sum")).reset_index()
    g["e_rb"] = g.rb / g.bb.clip(lower=1)
    g["e_br"] = g.br / g.bl.clip(lower=1)
    g["e_w"] = g.wk / g.bl.clip(lower=1)
    g["e_o"] = g.o / g.bb.clip(lower=1)
    d = d.merge(g[key + ["e_rb", "e_br", "e_w", "e_o"]], on=key, how="left")
    d["raa"] = d.runs_batter - d.e_rb * d.bat_balls
    d["exp_out"] = d.e_o * d.bat_balls
    d["bat_val"] = d.raa - w.wicket_value * (d.batter_out.astype(float) - d.exp_out)
    d["saved"] = d.e_br * d.bowl_balls - d.bowl_runs
    d["w_above"] = d.bowler_wkt.astype(float) - d.e_w * d.bowl_balls
    d["bowl_val"] = d.saved + w.wicket_value * d.w_above
    d["hi_lev"] = (d.over <= HIGH_LEVERAGE_OVERS[0]) | (d.over >= HIGH_LEVERAGE_OVERS[1])
    return d


def _fielding(d: pd.DataFrame, w: Weights) -> pd.DataFrame:
    """Catches, stumpings and run-outs per (match, player). Substitute fielders are ignored."""
    wk = d[d.is_dismissal & ~d.fielder_is_substitute].copy()
    rows = []
    c = wk[wk.wicket_type == "caught"]
    rows.append(pd.DataFrame({"match_id": c.match_id, "player": c.fielder_1, "team": c.bowl_team, "catches": 1, "stumpings": 0, "run_outs": 0.0}))
    cb = wk[wk.wicket_type == "caught and bowled"]     # no fielder listed: the bowler took it
    rows.append(pd.DataFrame({"match_id": cb.match_id, "player": cb.bowler, "team": cb.bowl_team, "catches": 1, "stumpings": 0, "run_outs": 0.0}))
    s = wk[wk.wicket_type == "stumped"]
    rows.append(pd.DataFrame({"match_id": s.match_id, "player": s.fielder_1, "team": s.bowl_team, "catches": 0, "stumpings": 1, "run_outs": 0.0}))
    ro = wk[wk.wicket_type == "run out"]
    for col in ["fielder_1", "fielder_2"]:
        part = ro[ro[col].notna()]
        share = 1.0 / ro.loc[part.index, ["fielder_1", "fielder_2"]].notna().sum(axis=1)
        rows.append(pd.DataFrame({"match_id": part.match_id, "player": part[col], "team": part.bowl_team, "catches": 0, "stumpings": 0, "run_outs": share}))
    f = pd.concat(rows, ignore_index=True).dropna(subset=["player"])
    f = f.groupby(["match_id", "player", "team"], as_index=False).sum()
    f["fielding_impact"] = w.catch * f.catches + w.stumping * f.stumpings + w.run_out * f.run_outs
    return f


def build_player_impact(d: pd.DataFrame, m: pd.DataFrame, xi: pd.DataFrame, w: Weights = Weights(),
                        runout_overrides: pd.DataFrame = None, keepers: pd.DataFrame = None) -> pd.DataFrame:
    """d = D.load_deliveries(), m = D.load_matches(), xi = D.load_xi(). Returns one row per player per match.
    runout_overrides: fielders for run-outs that Cricsheet leaves blank (see lib/runout_fix.py).
    keepers: (match_id, team, player) of listed wicketkeepers; adds a `wicketkeeper` column (blank where unknown)."""
    d = d.copy()
    if runout_overrides is not None and len(runout_overrides):
        k = ["match_id", "innings", "over", "ball"]
        o = runout_overrides.set_index(k)
        idx = d.set_index(k).index
        blank = (d.wicket_type == "run out") & d.fielder_1.isna()
        for col in ["fielder_1", "fielder_2"]:
            fill = pd.Series(idx.map(o[col]), index=d.index)
            d.loc[blank, col] = fill[blank]
    d = _ball_context(d, w)

    # ---- pressure flags per match ------------------------------------------------------------
    mm = m[["match_id", "winner", "stage", "win_by_runs", "win_by_wickets", "result_type", "chase_team",
            "player_of_match", "team1", "team2"]].copy()
    close = ((mm.win_by_runs <= CLOSE_RUNS) | (mm.win_by_wickets <= CLOSE_WKTS) | (mm.result_type == "super over"))
    mm["close_or_ko"] = (close.fillna(False) | (mm.stage != "league")).astype(bool)
    mm["chase_won"] = (mm.winner.notna() & (mm.winner == mm.chase_team))
    d = d.merge(mm[["match_id", "close_or_ko", "chase_won"]], on="match_id", how="left")
    d["close_or_ko"] = d.close_or_ko.fillna(False).astype(bool)
    d["chase_won"] = d.chase_won.fillna(False).astype(bool)
    d["is_chase_inn"] = d.innings == 2

    # ---- batting per (match, player) ---------------------------------------------------------
    d["_dotb"] = d.is_dot.astype(float)
    d["_bnd"] = (d.is_four | d.is_six).astype(float)
    d["_pp_bat"] = np.where(d.hi_lev, d.bat_val, 0.0)
    d["_close_bat"] = np.where(d.close_or_ko, d.bat_val, 0.0)
    d["_chase_bat"] = np.where(d.is_chase_inn & d.chase_won, d.bat_val, 0.0)
    bat = d.groupby(["match_id", "batter"]).agg(
        runs=("runs_batter", "sum"), balls_faced=("bat_balls", "sum"), fours=("is_four", "sum"), sixes=("is_six", "sum"),
        dismissed=("batter_out", "max"), runs_above_expected=("raa", "sum"), batting_impact=("bat_val", "sum"),
        _boundary_balls=("_bnd", "sum"), _hl=("_pp_bat", "sum"), _cl=("_close_bat", "sum"), _ch=("_chase_bat", "sum"),
    ).reset_index().rename(columns={"batter": "player"})

    # ---- bowling per (match, player) ---------------------------------------------------------
    d["_pp_bowl"] = np.where(d.hi_lev, d.bowl_val, 0.0)
    d["_close_bowl"] = np.where(d.close_or_ko, d.bowl_val, 0.0)
    d["_dot_b"] = d.is_dot_b.astype(float)
    bowl = d.groupby(["match_id", "bowler"]).agg(
        balls_bowled=("bowl_balls", "sum"), runs_conceded=("bowl_runs", "sum"), wickets=("bowler_wkt", "sum"),
        dots=("_dot_b", "sum"), bowling_impact=("bowl_val", "sum"),
        _hlb=("_pp_bowl", "sum"), _clb=("_close_bowl", "sum"),
    ).reset_index().rename(columns={"bowler": "player"})

    # match-average economy (both innings) for the "economy vs match average" field
    me = d.groupby("match_id").agg(_r=("bowl_runs", "sum"), _b=("bowl_balls", "sum"))
    me["match_economy"] = 6 * me._r / me._b.clip(lower=1)

    fld = _fielding(d, w)

    # ---- roster: playing XI + anyone who appears in the ball data --------------------------------
    r1 = xi[["match_id", "player", "team"]]
    r2 = d[["match_id", "batter", "bat_team"]].rename(columns={"batter": "player", "bat_team": "team"})
    r3 = d[["match_id", "bowler", "bowl_team"]].rename(columns={"bowler": "player", "bowl_team": "team"})
    r4 = fld[["match_id", "player", "team"]]        # fielders who are missing from the XI list
    roster = pd.concat([r1, r2, r3, r4], ignore_index=True).dropna().drop_duplicates(["match_id", "player"])
    roster = roster[roster.match_id.isin(mm.match_id)]

    t = roster.merge(bat, on=["match_id", "player"], how="left").merge(bowl, on=["match_id", "player"], how="left") \
              .merge(fld.drop(columns="team"), on=["match_id", "player"], how="left")
    t = t.merge(m[["match_id", "date", "season", "stage", "venue", "team1", "team2", "winner", "player_of_match"]],
                on="match_id", how="left")
    num = ["runs", "balls_faced", "fours", "sixes", "runs_above_expected", "batting_impact", "_boundary_balls", "_hl", "_cl", "_ch",
           "balls_bowled", "runs_conceded", "wickets", "dots", "bowling_impact", "_hlb", "_clb",
           "catches", "stumpings", "run_outs", "fielding_impact"]
    t[num] = t[num].fillna(0.0)
    t["dismissed"] = t.dismissed.fillna(False).astype(bool)
    t = t.merge(me[["match_economy"]], left_on="match_id", right_index=True, how="left")

    if keepers is not None and len(keepers):
        covered = set(keepers.match_id)
        kk = set(zip(keepers.match_id, keepers.player))
        t["wicketkeeper"] = [(1 if (a, b) in kk else 0) if a in covered else np.nan for a, b in zip(t.match_id, t.player)]
    else:
        t["wicketkeeper"] = np.nan
    t["opponent"] = np.where(t.team == t.team1, t.team2, t.team1)
    t["strike_rate"] = np.where(t.balls_faced > 0, 100 * t.runs / t.balls_faced.replace(0, np.nan), np.nan)
    t["boundary_pct"] = np.where(t.balls_faced > 0, 100 * t._boundary_balls / t.balls_faced.replace(0, np.nan), np.nan)
    t["economy"] = np.where(t.balls_bowled > 0, 6 * t.runs_conceded / t.balls_bowled.replace(0, np.nan), np.nan)
    t["econ_vs_match"] = t.economy - t.match_economy.where(t.balls_bowled > 0)
    t["dot_ball_pct"] = np.where(t.balls_bowled > 0, 100 * t.dots / t.balls_bowled.replace(0, np.nan), np.nan)
    t["team_win"] = (t.winner.notna() & (t.winner == t.team)).astype(int)
    t["player_of_match"] = (t.player == t.player_of_match).astype(int)

    # ---- pressure + win bonus ----------------------------------------------------------------
    # Impact earned in powerplay/death overs, in close or knockout matches, and in a successful chase
    # gets `pressure` extra weight each (stacking, so at most 1 + 3 x pressure on the same ball).
    # Impact is signed, so poor performances under pressure cost more too.
    t["pressure_impact"] = w.pressure * (t._hl + t._hlb + t._cl + t._clb + t._ch)
    involved = (t.balls_faced > 0) | (t.balls_bowled > 0) | (t.catches + t.stumpings + t.run_outs > 0)
    t["win_bonus"] = np.where((t.team_win == 1) & involved, w.win_bonus, 0.0)
    t["impact_score"] = t.batting_impact + t.bowling_impact + t.fielding_impact + t.pressure_impact + t.win_bonus

    t["catches"] = t.catches.astype(int)
    t["stumpings"] = t.stumpings.astype(int)
    t["run_outs"] = t.run_outs.round(2)
    for c in ["balls_faced", "fours", "sixes", "balls_bowled", "wickets"]:
        t[c] = t[c].astype(int)
    cols = ["match_id", "date", "season", "stage", "player", "team", "opponent",
            "runs", "balls_faced", "strike_rate", "fours", "sixes", "boundary_pct", "dismissed", "runs_above_expected",
            "wickets", "balls_bowled", "runs_conceded", "economy", "match_economy", "econ_vs_match", "dot_ball_pct",
            "catches", "stumpings", "run_outs", "wicketkeeper", "team_win", "player_of_match",
            "batting_impact", "bowling_impact", "fielding_impact", "pressure_impact", "win_bonus", "impact_score"]
    out = t[cols].sort_values(["date", "match_id", "impact_score"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def potm_validation(df: pd.DataFrame) -> dict:
    """How often does the Player of the Match come out on top of the impact score in that match?"""
    g = df.copy()
    g["rank"] = g.groupby("match_id").impact_score.rank(ascending=False, method="min")
    p = g[g.player_of_match == 1]
    if p.empty:
        return {"matches": 0, "top1": np.nan, "top3": np.nan, "avg_rank": np.nan}
    return {"matches": len(p), "top1": float((p["rank"] == 1).mean()), "top3": float((p["rank"] <= 3).mean()),
            "avg_rank": float(p["rank"].mean())}


def default_weights() -> dict:
    return asdict(Weights())
