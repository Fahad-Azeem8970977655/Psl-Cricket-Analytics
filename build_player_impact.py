"""Writes data/player_impact.csv (default weights).  Run:  python build_player_impact.py"""
from pathlib import Path
from lib import data as D
from lib.impact_match import build_player_impact, potm_validation, Weights
from lib.runout_fix import load_fixes

d, m, xi = D.load_deliveries(), D.load_matches(), D.load_xi()
ov, kp = load_fixes(d, xi, D.load_bio())
df = build_player_impact(d, m, xi, Weights(), runout_overrides=ov, keepers=kp)
out = Path(__file__).parent / "data" / "player_impact.csv"
df.assign(date=df.date.dt.strftime("%Y-%m-%d")).round(3).to_csv(out, index=False)
print(f"wrote {out}  rows={len(df):,}  matches={df.match_id.nunique()}  players={df.player.nunique()}")
print("Player-of-the-match check:", potm_validation(df))
