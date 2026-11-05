# PSL Analytics 2016-2026 (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```
First load takes a few seconds. Pages that train models (Win prediction, Luck vs skill) take 15-20 seconds the first time, then use the cache.

## Look and feel
Designed around cricket's own visual language rather than a generic dashboard kit: **Whites** page, **Turf** green primary, **Night** blue sidebar and scoreboard, **Leather** red for sixes and wickets, **Brass** scoreboard digits. Headings and numbers use Barlow Condensed (scoreboards are set in condensed type), body text Source Sans 3 (loaded from Google Fonts, so hard-refresh with Ctrl+F5 the first time). Key numbers appear as one scoreboard strip, the match centre opens with a real scoreboard, the overview opens with the ground graphic, and charts share one template. All of it lives in `lib/theme.py` and `lib/viz.py`; colours are at the top of `theme.py`, and the sidebar colours are in `.streamlit/config.toml`.

## Pages (16; related pages share a page and switch with the buttons at the top)
| Section | Page | What it does |
|---|---|---|
| Overview | Overview & data health | KPIs, champions, coverage, exact reconciliation with official totals, known issues, file inventory |
| | Seasons, champions & records | Finals, league heatmaps, awards check; records, extremes, super overs |
| Teams & matches | **Match centre** | Any of 358 matches: scorecards with how-out text, fall of wickets, worm, Manhattan, partnerships, phases, win probability, XI, umpires |
| | Team strength (Elo) | Sequential Elo, upsets, calibration |
| | Toss, chase & head-to-head | Binomial tests, H2H matrix |
| | Venues & phases | Scoring by ground; over x season heatmaps, pace vs spin, left vs right |
| Players | Batter scoring zones (pitch) | Ground + 22-yard pitch: dots, singles, twos, threes, fours, sixes; compare two batters, a team or the league |
| | **Partnerships & batting order** | Top stands, best pairs, by wicket, opening stands; runs and strike rate by position, who bats where, roles |
| | **Bowling & fielding** | Bowler profile; catches, run-outs, stumpings, team fielding, bowler/catcher combos |
| | Batter vs bowler | Matchups with empirical-Bayes shrinkage |
| | Player impact ratings | Runs above average by season/innings/over |
| | **Match impact score** | Batting + bowling + fielding + pressure + win bonus per player per match, adjustable weights, leaderboard, match explorer, player profile, CSV download |
| | Clusters, similar players & anomalies | KMeans/PCA, cosine similarity, Isolation Forest |
| Machine learning & honesty | Win prediction | Before the match (out-of-sample, near coin flip) and during the match (ball-by-ball, ~79% accurate) |
| | **Luck vs skill** | Coin-flip league simulations, season-to-season carry-over, split-half reliability of batting and bowling stats |
| Franchise | **Auction, squads & value for money (2026)** | Draft/auction, salary brackets, squad age, captains/coaches/owners, price vs 2026 impact, unmatched names, 2026 verification |

## Data (`/data`) and how it is used (`lib/data.py`)
- `deliveries_full.csv`, `matches_full.csv`, `match_id_map.csv`, `playing_xi.csv`, `players_master.csv`: exact Cricsheet-derived tables. Balls faced, overs, extras and fielders are exact. Runs, balls and wickets reconcile with `batting_season_stats.csv` / `bowling_season_stats.csv` in every season (Overview > Reconciliation).
- Super-over deliveries are excluded from all statistics; the 4 super-over winners come from `matches_fixed_results.csv`.
- The four 14-17 Nov 2020 playoff matches are labelled 2021 in the source and are corrected to 2020.
- Namesakes are joined by player id ("Mohammad Nawaz (3)" is the same person as "Mohammad Nawaz"; "Mohammad Imran (2)" is a different person).
- The Excel master workbook is not used: S2 (awards) and S5 (price brackets) disagree with the delivery data, and S3, S4 and S6 could not be verified (for example S3 lists Mark Chapman as Pakistani). Title counts come from `final_results.csv`.

## Derived file
`data/player_impact.csv` (7,951 rows, one per player per match) is built by `python build_player_impact.py` from `lib/impact_match.py` with default weights. It also uses two Cricbuzz-sourced files, `data/runout_fielders_cricbuzz.csv` (fielders for run-outs Cricsheet leaves blank) and `data/wicketkeepers_cricbuzz.csv`, resolved to dashboard names by `lib/runout_fix.py`. The dashboard recomputes it live from the sliders; the file is a snapshot for use elsewhere.

## Limits you should know
- **No shot direction in the data**, so the scoring-zone graphic is by *run type*, not a wagon wheel.
- Pre-match prediction is close to a coin flip, and the Luck vs skill page shows why (about 80% of the spread in league win rates is what luck alone produces).
- 2026 is mostly verified (all auction prices from one source, 7 of 8 coaches, 8 of 8 owners, 4 captains); Multan's coach is unconfirmed and 4 captains were not re-checked.
- Every match actually played has ball data: 358 matches. 357 come from Cricsheet; the 2016 Karachi Kings v Peshawar Zalmi match (11 Feb 2016) was rebuilt from ESPNcricinfo commentary (team totals match the scorecard, but two batters are 1 run high and extras read 8 and 11 instead of 9 and 12; umpires blank) and is left out of the reconciliation with the official season-stats file.
- Full bios (name, date of birth, nationality, role, batting hand, bowling style) exist for 503 of 514 players; the other 11 are 9 umpires/officials and 2 namesake ids whose copied bios were cleared. They come from a supplied workbook marked "Fully Verified" whose sources were not independently re-checked. Three CJ Green / PR Stirling / SD Hope values were corrected after an ESPNcricinfo check; two namesake Mohammad Irfan bios were cleared; 4 birth dates are 1 January (possible placeholders).
- Auction-to-performance joins use the reviewed name map (852 of 1,202 rows); 26 priced 2026 players are still unmatched.
- Fielding data has dismissals only (no dropped catches or misfields) and no wicketkeeper flag.
- Match impact score: Cricsheet names no fielder on 145 of 342 run-outs; 67 are filled from Cricbuzz (matched against that team's XI, 174/176 agreement on run-outs where both sources exist), so about 78 still have no fielder credited. Cricbuzz sources were not independently re-checked. Wicketkeeper flag covers 290 of 358 matches; the score is validated only loosely (Player of the Match is top-scored in ~50% of matches, top 3 in ~86%).
