"""Builds data/shot_locations.csv from the Cricinfo shot-location export (tracked balls only).

Usage:  python build_shot_locations.py path/to/PSL_2016_2026_Shot_Location_Data.csv

* x, y: ball landing point on the ground, centre = batter, 1.0 = boundary rope, bowler's end = +y.
* The source coordinates are absolute, so left-handers are mirrored: off side is always -x (left), leg side +x (right).
  Handedness is read from each batter's own off-side shots, not from the bio file.
* sector: 8 equal 45-degree wedges measured clockwise from straight back down the ground.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib import data as D

SECTORS = ["Mid on", "Mid wicket", "Square leg", "Fine leg", "Third man", "Point", "Cover", "Mid off"]
OFF = {"cover", "point", "mid off", "third man"}
SCALE = 180.5          # pixels per boundary radius (fitted: median 180.3, sd 3.6)
CENTRE = 181.5


def build(src: str) -> pd.DataFrame:
    a = pd.read_csv(src, low_memory=False)
    t = a[a.wagon_x.notna()].copy()
    t["batter"], t["bowler"] = D.canon(t.batter), D.canon(t.bowler)
    dx, dy = t.wagon_x - CENTRE, t.wagon_y - CENTRE
    off_dx = dx[t.shot_direction.isin(OFF)].groupby(t.batter).median()      # off side is on the left (-x) for a right-hander
    hand = pd.Series(np.where(off_dx < 0, "R", "L"), index=off_dx.index)
    t["batter_hand"] = t.batter.map(hand)
    t = t[t.batter_hand.notna()].drop(columns=["season"]).copy()
    dx = dx.loc[t.index]; dy = dy.loc[t.index]
    t["x"] = (np.where(t.batter_hand == "L", -dx, dx) / SCALE).round(4)
    t["y"] = (dy / SCALE).round(4)
    ang = np.degrees(np.arctan2(t.x, t.y)) % 360
    t["sector"] = [SECTORS[int(v // 45)] for v in ang]
    m = D.load_matches()[["match_id", "season"]].rename(columns={"season": "dash_season"})
    t = t.merge(m, on="match_id", how="inner")                              # dashboard season (fixes the Nov-2020 matches)
    bio = D.load_bio()
    t["bowl_type"] = t.bowler.map(dict(zip(bio.player, bio.bowl_type)))
    t["over"] = t.over_number + 1
    out = t.rename(columns={"dash_season": "season", "runs_scored": "runs", "wagon_distance": "dist"})
    cols = ["match_id", "season", "innings", "over", "batter", "batter_hand", "bowler", "bowl_type", "runs",
            "shot_played", "shot_control", "shot_direction", "x", "y", "dist", "sector"]
    return out[cols].sort_values(["season", "match_id", "innings", "over"]).reset_index(drop=True)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "PSL_2016_2026_Shot_Location_Data.csv"
    df = build(src)
    dest = Path(__file__).parent / "data" / "shot_locations.csv"
    df.to_csv(dest, index=False)
    print(f"wrote {dest}  rows={len(df):,}  batters={df.batter.nunique()}  seasons={sorted(df.season.unique())}")
