"""Fill run-out fielders that Cricsheet leaves blank, using the Cricbuzz-enriched run-out file.

Cricbuzz gives short names ("Wood", "M Rizwan", "Sarfaraz"). Each one is matched ONLY against the
fielding team's players in that match (playing XI + anyone seen in the ball data), and is accepted only
if exactly one player fits. Ambiguous, unknown and "(sub" (substitute) names are left blank.
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


def _tok(s) -> list:
    return [t for t in re.sub(r"[^a-z ]", " ", str(s).lower()).split() if t]


def _fits(cb_tokens, cand_tokens) -> bool:
    """Every Cricbuzz token must match a candidate token: exact, an initial, or a close spelling (Sarfaraz/Sarfraz)."""
    if not cb_tokens:
        return False
    for t in cb_tokens:
        if t in cand_tokens:
            continue
        if len(t) == 1 and any(c.startswith(t) for c in cand_tokens):
            continue
        if len(t) >= 5 and any(len(c) >= 5 and SequenceMatcher(None, t, c).ratio() >= 0.86 for c in cand_tokens):
            continue
        return False
    return True


def resolve(name, candidates: dict):
    """candidates: {dashboard_name: set(tokens)}. Returns a dashboard name, or None."""
    if not isinstance(name, str) or not name.strip() or "(sub" in name.lower() or name.lower().startswith("sub"):
        return None
    if name in candidates:
        return name
    cb = _tok(name)
    hits = [p for p, toks in candidates.items() if _fits(cb, toks)]
    if len(hits) == 1:
        return hits[0]
    # several fit: prefer the one whose full token set equals the Cricbuzz name
    exact = [p for p in hits if set(candidates[p]) == set(cb)]
    return exact[0] if len(exact) == 1 else None


def build_overrides(path, d: pd.DataFrame, xi: pd.DataFrame, bio: pd.DataFrame):
    """Returns (overrides, report). overrides is indexed by (match_id, innings, over, ball) with fielder_1 / fielder_2."""
    r = pd.read_csv(path, encoding="utf-8-sig")
    r["over"] = r.over_number + 1                      # the export counts overs from 0
    r = r.rename(columns={"ball_number": "ball"})
    extra = {}
    if {"full_name", "player"}.issubset(bio.columns):
        extra = {p: _tok(f) for p, f in zip(bio.player, bio.full_name.fillna(""))}
        if "aliases" in bio.columns:
            for p, a in zip(bio.player, bio.aliases.fillna("")):
                extra[p] = extra.get(p, []) + _tok(a)

    roster = pd.concat([xi[["match_id", "team", "player"]],
                        d[["match_id", "bowl_team", "bowler"]].rename(columns={"bowl_team": "team", "bowler": "player"})]
                       ).dropna().drop_duplicates()
    by_team = {k: g.player.tolist() for k, g in roster.groupby(["match_id", "team"])}

    has = d[d.wicket_type == "run out"].set_index(["match_id", "innings", "over", "ball"])
    rows, stats = [], dict(blank_in_source=0, filled=0, still_blank=0, unresolved=0, sub=0)
    for x in r.itertuples():
        key = (x.match_id, x.innings, x.over, x.ball)
        if key not in has.index:
            continue
        cur = has.loc[key]
        cur = cur.iloc[0] if isinstance(cur, pd.DataFrame) else cur
        if pd.notna(cur.fielder_1):
            continue                                    # Cricsheet already has it: never overwrite
        stats["blank_in_source"] += 1
        cands = {p: set(_tok(p)) | set(extra.get(p, [])) for p in by_team.get((x.match_id, x.fielding_team), [])}
        names = [n for n in [x.fielder_responsible, x.additional_fielder] if isinstance(n, str)]
        if not names:
            stats["still_blank"] += 1
            continue
        res = [resolve(n, cands) for n in names]
        subs = sum(1 for n in names if "(sub" in n.lower())
        stats["sub"] += subs
        good = [p for p in res if p]
        if not good:
            stats["unresolved"] += (len(names) - subs)
            continue
        stats["unresolved"] += sum(1 for n, p in zip(names, res) if p is None and "(sub" not in n.lower())
        stats["filled"] += 1
        rows.append({"match_id": x.match_id, "innings": x.innings, "over": x.over, "ball": x.ball,
                     "fielder_1": good[0], "fielder_2": good[1] if len(good) > 1 else np.nan})
    ov = pd.DataFrame(rows)
    return ov, stats


def validate(path, d: pd.DataFrame, xi: pd.DataFrame, bio: pd.DataFrame) -> dict:
    """Accuracy check on run-outs where Cricsheet ALREADY names the fielder and Cricbuzz also gives a name."""
    r = pd.read_csv(path, encoding="utf-8-sig")
    r["over"] = r.over_number + 1
    has = d[(d.wicket_type == "run out") & d.fielder_1.notna()].set_index(["match_id", "innings", "over", "ball"])
    roster = pd.concat([xi[["match_id", "team", "player"]],
                        d[["match_id", "bowl_team", "bowler"]].rename(columns={"bowl_team": "team", "bowler": "player"})]).dropna().drop_duplicates()
    by_team = {k: g.player.tolist() for k, g in roster.groupby(["match_id", "team"])}
    ok = tot = 0
    for x in r.itertuples():
        key = (x.match_id, x.innings, x.over, x.ball_number)
        if key not in has.index or not isinstance(x.fielder_responsible, str):
            continue
        cur = has.loc[key]; cur = cur.iloc[0] if isinstance(cur, pd.DataFrame) else cur
        cands = {p: set(_tok(p)) for p in by_team.get((x.match_id, x.fielding_team), [])}
        res = resolve(x.fielder_responsible, cands)
        if res is None:
            continue
        tot += 1
        ok += res in {cur.fielder_1, cur.fielder_2}
    return {"checked": tot, "agree": ok}


def build_keepers(path, d: pd.DataFrame, xi: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """Wicketkeeper per (match, team) from the Cricbuzz file, resolved to dashboard names. Unresolved rows are dropped."""
    k = pd.read_csv(path, encoding="utf-8-sig")
    extra = {p: _tok(f) for p, f in zip(bio.player, bio.full_name.fillna(""))} if "full_name" in bio.columns else {}
    roster = pd.concat([xi[["match_id", "team", "player"]],
                        d[["match_id", "bat_team", "batter"]].rename(columns={"bat_team": "team", "batter": "player"})]).dropna().drop_duplicates()
    by_team = {kk: g.player.tolist() for kk, g in roster.groupby(["match_id", "team"])}
    out = []
    for x in k.itertuples():
        cands = {p: set(_tok(p)) | set(extra.get(p, [])) for p in by_team.get((x.match_id, x.team), [])}
        p = resolve(x.wicketkeeper, cands)
        if p:
            out.append({"match_id": x.match_id, "team": x.team, "player": p})
    return pd.DataFrame(out).drop_duplicates()


def load_fixes(d: pd.DataFrame, xi: pd.DataFrame, bio: pd.DataFrame):
    """(run-out overrides, keepers) built from the files in /data, or (None, None) if they are not there."""
    from pathlib import Path
    data = Path(__file__).resolve().parent.parent / "data"
    f1, f2 = data / "runout_fielders_cricbuzz.csv", data / "wicketkeepers_cricbuzz.csv"
    ov = build_overrides(f1, d, xi, bio)[0] if f1.exists() else None
    kp = build_keepers(f2, d, xi, bio) if f2.exists() else None
    return ov, kp
