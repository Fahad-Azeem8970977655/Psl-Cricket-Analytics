"""Data layer for the PSL dashboard: loading, cleaning, joins and cached summaries.

Sources (all in /data):
* deliveries_full.csv  - one row per delivery with wides/no-balls/byes/leg-byes, fielders, super-over flag.
* matches_full.csv + match_id_map.csv - one row per match (358); map gives the row in the older table.
* players_master.csv   - Cricsheet player_id <-> names (+ bios for 120 players).
* playing_xi.csv       - the players listed for each team in each match.

Rules applied here (each was checked against the files):
* Super-over deliveries are excluded from all statistics (kept separately for the match centre).
* The four 14-17 Nov 2020 playoff matches are labelled season 2021 in the source; they belong to 2020.
* Namesakes are joined by player_id: "Mohammad Nawaz" and "Mohammad Nawaz (3)" share one id (one person);
  "Mohammad Imran (2)" has its own id (a different person).
* Matches Cricsheet records as a tie settled by a super over get their winner from matches_fixed_results.csv.
* Balls faced = deliveries that are not wides (no-balls count); bowler runs = bat runs + wides + no-balls.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"

TEAM_COLORS = {
    "Islamabad United": "#d62728",
    "Karachi Kings": "#1f77b4",
    "Lahore Qalandars": "#2ca02c",
    "Multan Sultans": "#17becf",
    "Peshawar Zalmi": "#d4a017",
    "Quetta Gladiators": "#7b3fa0",
    "Hyderabad Kingsmen": "#ff7f0e",
    "Rawalpindiz": "#8c564b",
}
TEAMS = list(TEAM_COLORS)
# Matches added by hand from ESPNcricinfo (not in Cricsheet, not in the official season-stats file)
ADDED_MATCH_IDS = {959193}   # 2016 Karachi Kings v Peshawar Zalmi
PHASES = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
NON_WICKET = {"retired hurt", "retired not out"}
NOT_BOWLER = {"run out", "retired out", "obstructing the field"}
VENUE_FIX = {
    "National Stadium, Karachi": "National Stadium",
    "Gaddafi Stadium, Lahore": "Gaddafi Stadium",
    "Sheikh Zayed Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig", low_memory=False)


def phase_of(over):
    over = np.asarray(over)
    return np.select([over <= 6, over <= 15], PHASES[:2], PHASES[2])


# --------------------------------------------------------------------------- players
@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    """One row per player_id. The canonical name is the row that has a bio, else the one without a '(n)' suffix."""
    pm = _csv("players_master.csv")
    pm["_bio"] = pm.full_name.notna()
    pm["_style"] = pm.batting_style.notna() | pm.bowling_style.notna()
    pm["_suf"] = pm.cricsheet_name.str.contains(r"\(\d+\)$")
    pm = pm.sort_values(["player_id", "_bio", "_style", "_suf", "cricsheet_name"], ascending=[True, False, False, True, True])
    names = pm.groupby("player_id").cricsheet_name.apply(list)
    one = pm.drop_duplicates("player_id").copy()
    for c in ("full_name", "date_of_birth", "nationality", "role", "batting_style", "bowling_style"):   # fill from any spelling of the same id
        one[c] = one[c].fillna(one.player_id.map(pm.groupby("player_id")[c].first()))
    one["player"] = one.cricsheet_name
    one["aliases"] = one.player_id.map(names)
    one["dob"] = pd.to_datetime(one.date_of_birth, errors="coerce")
    one["bowl_type"] = one.bowling_style.map(_bowl_type)
    return one.drop(columns=["_bio", "_style", "_suf"]).reset_index(drop=True)


def _bowl_type(s):
    s = str(s).lower()
    if "fast" in s or "medium" in s:
        return "Pace"
    if any(k in s for k in ("break", "orthodox", "spin", "chinaman", "unorthodox")):
        return "Spin"
    return "Unknown"


@st.cache_data(show_spinner=False)
def alias_map() -> dict:
    """any spelling used in the source files -> canonical name (joined by player_id)."""
    pm = _csv("players_master.csv")
    canon = load_master().set_index("player_id").player
    return dict(zip(pm.cricsheet_name, pm.player_id.map(canon)))


@st.cache_data(show_spinner=False)
def id_map() -> dict:
    """canonical name -> player_id"""
    m = load_master()
    return dict(zip(m.player, m.player_id))


def canon(s: pd.Series) -> pd.Series:
    return s.map(alias_map()).fillna(s)


@st.cache_data(show_spinner=False)
def load_bio() -> pd.DataFrame:
    """Players with a full bio (name, date of birth, nationality, role). Umpires and match officials are not in the file."""
    b = load_master()
    return b[b.full_name.notna()].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_styles() -> pd.DataFrame:
    """Players with a known batting hand and/or bowling style (a larger set than the full bios)."""
    b = load_master()
    return b[b.batting_style.notna() | b.bowling_style.notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- matches
@st.cache_data(show_spinner=False)
def _raw_deliveries() -> pd.DataFrame:
    return _csv("deliveries_full.csv")


def _fix_season(m: pd.DataFrame) -> pd.DataFrame:
    late2020 = (m.date.dt.year == 2020) & (m.date.dt.month >= 10)
    m.loc[late2020, "season"] = 2020
    return m


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    """358 matches that have ball-by-ball data."""
    m = _csv("matches_full.csv")
    mp = _csv("match_id_map.csv")[["match_id", "matches_csv_row"]].rename(columns={"matches_csv_row": "match_idx"})
    m = m.merge(mp, on="match_id", how="left")
    m["date"] = pd.to_datetime(m["date"])
    m = _fix_season(m)
    m["venue"] = m["venue"].replace(VENUE_FIX)
    m["stage"] = m.stage.replace({"group": "league"})
    m["player_of_match"] = canon(m.player_of_match)
    m["result_type"] = np.where(m.super_over.astype(bool), "super over", m.result)

    fx = _csv("matches_fixed_results.csv")
    fx["date"] = pd.to_datetime(fx["date"])
    for _, r in fx[fx.true_result_type == "super over"].iterrows():
        k = (m.date == r.date) & (((m.team1 == r.team1) & (m.team2 == r.team2)) |
                                  ((m.team1 == r.team2) & (m.team2 == r.team1)))
        m.loc[k, "winner"] = r.winner

    d = _raw_deliveries()
    d = d[~d.is_super_over]
    first = d[d.innings == 1].groupby("match_id").batting_team.first().rename("bat_first")
    m = m.merge(first, on="match_id", how="left")
    m["chase_team"] = np.where(m.bat_first == m.team1, m.team2, m.team1)
    wk = d.wicket_kind.notna() & ~d.wicket_kind.isin(NON_WICKET)
    t = d.assign(wk=wk).groupby(["match_id", "innings"]).agg(runs=("total_runs", "sum"), wkts=("wk", "sum")).unstack("innings")
    t.columns = [f"inn{i}_{c}" for c, i in t.columns]
    m = m.merge(t.reset_index(), on="match_id", how="left")
    nb = ~d.non_boundary.astype(bool)
    bm = pd.DataFrame({"fours": ((d.runs_batter == 4) & nb).groupby(d.match_id).sum(),
                       "sixes": ((d.runs_batter == 6) & nb).groupby(d.match_id).sum()}).reset_index()
    m = m.merge(bm, on="match_id", how="left")

    m["decided"] = m.winner.notna()
    m["total_runs"] = m.inn1_runs + m.inn2_runs
    m["chase_won"] = m.decided & (m.winner == m.chase_team)
    m["bat_first_won"] = m.decided & (m.winner == m.bat_first)
    m["toss_winner_won"] = m.decided & (m.winner == m.toss_winner)
    m["margin_type"] = np.select(
        [m.win_by_runs.notna(), m.win_by_wickets.notna(), m.result_type == "super over"],
        ["runs", "wickets", "super over"], "-")
    m["margin_value"] = m.win_by_runs.fillna(m.win_by_wickets)
    return m.sort_values("match_idx").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_matches_all() -> pd.DataFrame:
    """Matches + the played fixture(s) missing from Cricsheet (no ball data) - used for Elo."""
    m = load_matches().copy()
    mm = _csv("missing_matches.csv")
    mm = mm[mm.match_status == "played"]
    add = pd.DataFrame({
        "season": mm.season, "date": pd.to_datetime(mm.date), "venue": mm.venue.replace(VENUE_FIX),
        "team1": mm.team1, "team2": mm.team2, "toss_winner": mm.toss_winner,
        "toss_decision": mm.toss_decision, "winner": mm.winner, "result_type": "normal",
        "stage": "league", "player_of_match": mm.player_of_match, "decided": mm.winner.notna(),
    })
    other = np.where(add.toss_winner == add.team1, add.team2, add.team1)
    add["bat_first"] = np.where(add.toss_decision == "bat", add.toss_winner, np.where(add.toss_decision == "field", other, np.nan))
    add["chase_team"] = np.where(add.bat_first == add.team1, add.team2, np.where(add.bat_first == add.team2, add.team1, np.nan))
    out = pd.concat([m, add], ignore_index=True).sort_values(["date", "match_idx"], kind="stable")
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_xi() -> pd.DataFrame:
    xi = _csv("playing_xi.csv")
    m = load_matches()[["match_id", "season", "date", "match_idx"]]
    xi = xi.drop(columns=["season"]).merge(m, on="match_id", how="left")
    xi["player"] = canon(xi.player)
    return xi.drop(columns=["cricinfo_id", "source_url"], errors="ignore")


@st.cache_data(show_spinner=False)
def load_squads() -> pd.DataFrame:
    xi = load_xi()
    return xi.groupby(["season", "team", "player"]).size().rename("matches").reset_index()


# --------------------------------------------------------------------------- deliveries
def _prepare(b: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    b = b.drop(columns=["season", "source_url"], errors="ignore").merge(m, on="match_id", how="left")
    b["date"] = pd.to_datetime(b["date"])
    for c in ["batter", "bowler", "non_striker", "player_out", "fielder_1", "fielder_2"]:
        b[c] = canon(b[c])
    b = b.rename(columns={"batting_team": "bat_team", "bowling_team": "bowl_team", "delivery_in_over": "ball",
                          "extras_total": "extras", "wicket_kind": "wicket_type", "total_runs": "total"})
    b = b.sort_values(["match_idx", "innings", "over", "ball"], kind="stable").reset_index(drop=True)
    nb = ~b.non_boundary.astype(bool)
    b["fielder_is_substitute"] = b.fielder_is_substitute.eq(True)
    b["is_legal"] = b.is_legal.astype(bool)
    b["is_dismissal"] = b.wicket_type.notna() & ~b.wicket_type.isin(NON_WICKET)
    b["bowler_wkt"] = b.is_dismissal & ~b.wicket_type.isin(NOT_BOWLER)
    b["batter_out"] = b.is_dismissal & (b.player_out == b.batter)
    b["bat_balls"] = (b.wides == 0).astype(float)          # batters face no-balls, not wides
    b["bowl_balls"] = b.is_legal.astype(float)
    b["bowl_runs"] = b.runs_batter + b.wides + b.noballs   # byes, leg-byes and penalties are not the bowler's
    b["is_dot"] = (b.wides == 0) & (b.runs_batter == 0)
    b["is_dot_b"] = b.is_legal & (b.bowl_runs == 0)
    b["is_single"] = b.runs_batter == 1
    b["is_two"] = b.runs_batter == 2
    b["is_three"] = b.runs_batter == 3
    b["is_four"] = (b.runs_batter == 4) & nb
    b["is_six"] = (b.runs_batter == 6) & nb
    b["phase"] = phase_of(b.over)
    return b


@st.cache_data(show_spinner="Preparing ball-by-ball data...")
def load_deliveries() -> pd.DataFrame:
    m = load_matches()[["match_id", "match_idx", "season", "venue", "city", "stage", "bat_first"]]
    b = _raw_deliveries()
    b = _prepare(b[~b.is_super_over].copy(), m)
    st_ = load_styles()
    b["bowl_type"] = b.bowler.map(dict(zip(st_.player, st_.bowl_type))).fillna("Unknown")
    b["bat_hand"] = b.batter.map(dict(zip(st_.player, st_.batting_style))).fillna("Unknown")
    return b


@st.cache_data(show_spinner=False)
def load_super_over_deliveries() -> pd.DataFrame:
    m = load_matches()[["match_id", "match_idx", "season", "venue", "city", "stage", "bat_first"]]
    b = _raw_deliveries()
    return _prepare(b[b.is_super_over].copy(), m)


# --------------------------------------------------------------------------- summaries
def _outs(d: pd.DataFrame, by: str) -> pd.Series:
    dis = d[d.is_dismissal]
    return dis.groupby("player_out" if by == "batter" else by).size().rename("outs")


def batting_summary(d: pd.DataFrame, by: str = "batter") -> pd.DataFrame:
    g = d.groupby(by).agg(
        runs=("runs_batter", "sum"), balls=("bat_balls", "sum"), fours=("is_four", "sum"),
        sixes=("is_six", "sum"), singles=("is_single", "sum"), twos=("is_two", "sum"),
        threes=("is_three", "sum"), dots=("is_dot", "sum"), innings=("match_idx", "nunique"),
    ).reset_index()
    g = g.merge(_outs(d, by), left_on=by, right_index=True, how="left")
    g["outs"] = g.outs.fillna(0)
    g["balls"] = g.balls.round(0)
    g["sr"] = 100 * g.runs / g.balls.replace(0, np.nan)
    g["avg"] = g.runs / g.outs.replace(0, np.nan)
    g["boundary_pct"] = 100 * (g.fours + g.sixes) / g.balls.replace(0, np.nan)
    g["dot_pct"] = 100 * g.dots / g.balls.replace(0, np.nan)
    g["boundary_runs_pct"] = 100 * (4 * g.fours + 6 * g.sixes) / g.runs.replace(0, np.nan)
    g["balls_per_boundary"] = g.balls / (g.fours + g.sixes).replace(0, np.nan)
    return g


def bowling_summary(d: pd.DataFrame, by: str = "bowler") -> pd.DataFrame:
    g = d.groupby(by).agg(
        balls=("bowl_balls", "sum"), runs=("bowl_runs", "sum"), wkts=("bowler_wkt", "sum"),
        dots=("is_dot_b", "sum"), fours=("is_four", "sum"), sixes=("is_six", "sum"),
        matches=("match_idx", "nunique"),
    ).reset_index()
    g["balls"] = g.balls.round(0)
    g["overs"] = g.balls / 6
    g["economy"] = g.runs / g.overs.replace(0, np.nan)
    g["bowl_sr"] = g.balls / g.wkts.replace(0, np.nan)
    g["bowl_avg"] = g.runs / g.wkts.replace(0, np.nan)
    g["dot_pct"] = 100 * g.dots / g.balls.replace(0, np.nan)
    g["boundary_pct"] = 100 * (g.fours + g.sixes) / g.balls.replace(0, np.nan)
    return g


def batter_innings(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["match_idx", "season", "bat_team", "batter"]).agg(
        runs=("runs_batter", "sum"), balls=("bat_balls", "sum"),
        fours=("is_four", "sum"), sixes=("is_six", "sum")).reset_index()
    out = d[d.is_dismissal][["match_idx", "player_out"]].drop_duplicates()
    out = out.rename(columns={"player_out": "batter"}).assign(out=1)
    g = g.merge(out, on=["match_idx", "batter"], how="left")
    g["out"] = g.out.fillna(0).astype(int)
    g["balls"] = g.balls.round(0)
    return g


def run_mix(d: pd.DataFrame) -> dict:
    """Counts used by the field/pitch graphic."""
    return dict(
        runs=int(d.runs_batter.sum()), balls=int(round(d.bat_balls.sum())),
        dots=int(d.is_dot.sum()), singles=int(d.is_single.sum()), twos=int(d.is_two.sum()),
        threes=int(d.is_three.sum()), fours=int(d.is_four.sum()), sixes=int(d.is_six.sum()))


# --------------------------------------------------------------------------- other tables
@st.cache_data(show_spinner=False)
def load_points() -> pd.DataFrame:
    return _csv("points_table_official.csv")


@st.cache_data(show_spinner=False)
def load_final_results() -> pd.DataFrame:
    f = _csv("final_results.csv")
    f["final_date"] = pd.to_datetime(f.final_date)
    num = f.margin.str.extract(r"(\d+)\s*(runs?|wickets?)", expand=True)
    f["margin_value"] = pd.to_numeric(num[0], errors="coerce")
    f["margin_type"] = num[1].str.rstrip("s")
    return f


@st.cache_data(show_spinner=False)
def load_third_fourth() -> pd.DataFrame:
    return _csv("third_fourth_place.csv")


@st.cache_data(show_spinner=False)
def load_awards() -> pd.DataFrame:
    return _csv("season_awards_check.csv")


@st.cache_data(show_spinner=False)
def load_team_seasons() -> pd.DataFrame:
    return _csv("team_seasons_final2.csv")


@st.cache_data(show_spinner=False)
def load_auction() -> pd.DataFrame:
    """Draft/auction rows + the reviewed name map (cricsheet name / player_id) + extra nationalities."""
    a = _csv("auction_draft_final2.csv").reset_index(drop=True)
    a["auction_row"] = a.index
    fill = _csv("auction_nationality_fill.csv")[["auction_row", "nationality"]].rename(columns={"nationality": "nat_fill"})
    a = a.merge(fill, on="auction_row", how="left")
    a["nationality"] = a.nationality.fillna(a.nat_fill)
    mp = _csv("auction_name_map.csv")[["auction_row", "cricsheet_name", "player_id", "match_method", "confidence"]]
    a = a.drop(columns=["cricsheet_name", "player_id"], errors="ignore").merge(mp, on="auction_row", how="left")
    a["canonical"] = canon(a.cricsheet_name)
    return a.drop(columns=["nat_fill"])


@st.cache_data(show_spinner=False)
def load_verification() -> pd.DataFrame:
    return _csv("verification_2026.csv")


@st.cache_data(show_spinner=False)
def load_category_prices() -> pd.DataFrame:
    p = _csv("auction_category_prices_final.csv")
    nums = p.price.fillna("").map(lambda s: [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", s)])
    p["usd_low"] = [n[0] if n and "maximum" not in str(s).lower() else np.nan for n, s in zip(nums, p.price)]
    p["usd_high"] = [n[-1] if n else np.nan for n in nums]
    return p


@st.cache_data(show_spinner=False)
def load_batting_stats() -> pd.DataFrame:
    return _csv("batting_season_stats.csv")


@st.cache_data(show_spinner=False)
def load_bowling_stats() -> pd.DataFrame:
    return _csv("bowling_season_stats.csv")


@st.cache_data(show_spinner=False)
def load_excel() -> dict:
    """S3 (platinum players), S4 (champion squads), S6 (title counts + records) from the master workbook.
    S2 (awards) and S5 (price brackets) are intentionally NOT used: they disagree with the delivery data."""
    x = pd.ExcelFile(DATA / "HBL_PSL_11Year_Master_2016_2026.xlsx")
    find = lambda p: [s for s in x.sheet_names if s.startswith(p)][0]
    s3 = x.parse(find("S3"), header=1)
    s3["base_cr"] = pd.to_numeric(s3["Base Salary (PKR Crore)"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    s4 = x.parse(find("S4"), header=1)
    s6 = x.parse(find("S6"), header=None)
    try:
        a = s6.iloc[3:11, 0:7].copy(); a.columns = s6.iloc[2, 0:7].tolist()
        r = s6.iloc[14:26, 0:5].copy(); r.columns = s6.iloc[13, 0:5].tolist()
    except Exception:                                   # layout changed
        a, r = pd.DataFrame(), pd.DataFrame()
    return {"platinum": s3, "champion_squads": s4, "titles": a.reset_index(drop=True), "records": r.reset_index(drop=True)}


# --------------------------------------------------------------------------- reconciliation
@st.cache_data(show_spinner=False)
def reconciliation() -> dict:
    """Delivery totals (regular + super over) vs the official season stats files."""
    reg, so = load_deliveries(), load_super_over_deliveries()
    both = pd.concat([reg, so])
    both = both[~both.match_id.isin(ADDED_MATCH_IDS)]      # the official season-stats file does not contain the added match
    bat = load_batting_stats().assign(player=lambda x: canon(x.player))
    bowl = load_bowling_stats().assign(player=lambda x: canon(x.player))
    mine_bat = both.groupby(["season", "batter"]).agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum")).reset_index().rename(columns={"batter": "player"})
    off_bat = bat.groupby(["season", "player"])[["runs", "balls"]].sum().reset_index()
    cb = off_bat.merge(mine_bat, on=["season", "player"], how="outer", suffixes=("_official", "_deliveries")).fillna(0)
    mine_bowl = both.groupby(["season", "bowler"]).agg(wkts=("bowler_wkt", "sum"), runs=("bowl_runs", "sum")).reset_index().rename(columns={"bowler": "player"})
    off_bowl = bowl.groupby(["season", "player"])[["wickets", "runs_conceded"]].sum().reset_index()
    cw = off_bowl.merge(mine_bowl, on=["season", "player"], how="outer").fillna(0)
    season = cb.groupby("season")[["runs_official", "runs_deliveries", "balls_official", "balls_deliveries"]].sum()
    season["runs_diff"] = season.runs_official - season.runs_deliveries
    season["balls_diff"] = season.balls_official - season.balls_deliveries
    season["wickets_official"] = cw.groupby("season").wickets.sum()
    season["wickets_deliveries"] = cw.groupby("season").wkts.sum()
    season["wickets_diff"] = season.wickets_official - season.wickets_deliveries
    n_bad = int(((cb.runs_official != cb.runs_deliveries) | (cb.balls_official != cb.balls_deliveries)).sum())
    return {"season": season.reset_index(), "player_seasons": len(cb), "player_seasons_off": n_bad}


@st.cache_data(show_spinner=False)
def lineage_check() -> dict:
    m, mp = load_matches(), _csv("match_id_map.csv")
    d = _raw_deliveries()
    xi = _csv("playing_xi.csv")
    return {
        "matches": len(m), "deliveries": len(d), "super_over_rows": int(d.is_super_over.sum()),
        "ids_agree": set(m.match_id) == set(d.match_id) == set(mp.match_id) == set(xi.match_id),
        "extras_ok": int(((d.wides + d.noballs + d.byes + d.legbyes + d.penalty) != d.extras_total).sum()) == 0,
    }


def inventory() -> pd.DataFrame:
    rows = []
    for f in sorted(DATA.iterdir()):
        n = len(pd.read_csv(f, encoding="utf-8-sig", low_memory=False)) if f.suffix == ".csv" else None
        rows.append({"file": f.name, "rows": n, "size_kb": round(f.stat().st_size / 1024, 1)})
    return pd.DataFrame(rows)
