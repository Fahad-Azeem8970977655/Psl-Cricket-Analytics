import pandas as pd
import plotly.express as px
import streamlit as st

from lib import data as D, ui


def render():
    seasons = ui.selected_seasons()
    m = D.load_matches()
    d = D.load_deliveries()
    fr = D.load_final_results()
    ms, ds = m[m.season.isin(seasons)], d[d.season.isin(seasons)]

    ui.kpis([("Seasons", len(seasons)), ("Matches with ball data", len(ms)), ("Deliveries", f"{len(ds):,}"),
             ("Players", len(set(ds.batter) | set(ds.bowler))), ("Venues", ms.venue.nunique()),
             ("Runs scored", f"{int(ds.total.sum()):,}")])

    t1, t2, t3, t4, t5 = st.tabs(["Champions", "Coverage", "Reconciliation", "Known issues", "Files"])

    with t1:
        f = fr[fr.season.isin(seasons)]
        titles = f.champion.value_counts().rename_axis("team").reset_index(name="titles")
        fin = pd.concat([f.champion, f.runner_up]).value_counts().rename_axis("team").reset_index(name="finals")
        both = titles.merge(fin, on="team", how="outer").fillna(0).sort_values(["titles", "finals"], ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(both.melt("team", var_name="what", value_name="count"), x="team", y="count", color="what",
                         barmode="group", color_discrete_sequence=["#1F6B45", "#C9962E"], title="Titles and finals played")
            ui.show(fig)
        with c2:
            ui.table(f[["season", "champion", "runner_up", "margin", "final_venue"]])
        st.caption("Champions, finals and margins come from `final_results.csv`, cross-checked against the finals in `matches_full.csv`.")

    with t2:
        pt = D.load_points()
        sched = (pt.groupby("season").played.sum() / 2).round().rename("scheduled (points table)")
        have = m.groupby("season").size().rename("with ball data")
        cov = pd.concat([sched, have], axis=1).reset_index()
        cov["not in ball data"] = cov["scheduled (points table)"] - cov["with ball data"]
        ui.table(cov)
        st.caption("League matches are counted from the official points table, so small differences are expected. "
                   "Cricsheet lacked 7 fixtures: 6 were abandoned without a ball bowled, and 1 was played (Karachi Kings v Peshawar Zalmi, 11 Feb 2016). "
                   "That one was rebuilt ball by ball from ESPNcricinfo commentary and is included, so every match actually played (358) now has ball data.")

    with t3:
        rc = D.reconciliation()
        lc = D.lineage_check()
        ui.kpis([("Matches", lc["matches"]), ("Deliveries", f"{lc['deliveries']:,}"), ("Super-over rows", lc["super_over_rows"]),
                 ("Player-seasons that differ from official stats", f"{rc['player_seasons_off']} of {rc['player_seasons']}")])
        ui.table(rc["season"].round(0))
        st.caption("Official season stats = deliveries in regular play + super overs. Runs, balls faced and wickets agree in every season, "
                   "which also confirms the 4 super-over matches and the November 2020 playoffs being counted in 2020. "
                   "The 2016 Karachi v Peshawar match added from ESPNcricinfo is left out of this comparison because the official season-stats file does not contain it.")
        s = D.batting_summary(d[(d.season == 2022) & (d.batter == "Fakhar Zaman")])
        stat = D.load_batting_stats().query("season == 2022 and player == 'Fakhar Zaman'")
        if len(s) and len(stat):
            st.metric("Fakhar Zaman 2022 runs (super over excluded)", int(s.runs.iloc[0]),
                      delta=f"source file said {int(stat.runs.iloc[0])}", delta_color="off")
            st.caption("588 is the official record. The source file said 590 because it included 2 runs from a super over.")
        st.caption(f"Files agree on match ids: {lc['ids_agree']}. Extras parts add up to the extras total on every delivery: {lc['extras_ok']}.")

    with t4:
        ui.md("""
        - **Exact ball data** (wides, no-balls, byes, leg-byes, fielders) comes from the Cricsheet JSON re-extract, so balls faced and overs are exact.
        - **Super overs** are excluded from all statistics. The 4 super-over winners come from `matches_fixed_results.csv` (the new match file leaves them blank).
        - **The 4 playoff matches of 14-17 Nov 2020** are labelled season 2021 in the source and are corrected to 2020.
        - **Namesakes are joined by player id**: "Mohammad Nawaz" and "Mohammad Nawaz (3)" are one person; "Mohammad Imran (2)" is a different person.
        - **The Excel master workbook is not used for results**: its awards (S2) and price brackets (S5) disagree with the delivery data, and its platinum list (S3), champion squads (S4) and records (S6) could not be verified (S3 lists Mark Chapman as Pakistani).
        - **2026 data is mostly, not fully, verified**: all 132 auction prices (one source, pcb.com.pk), 4 captains, 7 of 8 coaches and 8 of 8 owners (two sources each; Peshawar's coach and Lahore's owner were corrected). Multan's coach is unconfirmed and the other 4 captains were not re-checked.
        - **Shot direction is not in the data**, so the pitch graphic shows run *type*, not a wagon wheel.
        - **One match was added by hand**: 2016 Karachi Kings v Peshawar Zalmi (11 Feb 2016) is not in Cricsheet and was rebuilt from ESPNcricinfo commentary. Team totals, wickets, balls faced and fours/sixes match the scorecard, but Tamim Iqbal (38 v 37) and Nauman Anwar (15 v 14) are 1 run high and extras read 8 and 11 instead of 9 and 12. Umpires and referee are not recorded.
        - **Full bios (name, date of birth, nationality, role, batting hand, bowling style) cover 503 of 514 players**, from a supplied bio workbook (Cricbuzz / Wikipedia links in `bio_source_url`). It is labelled "Fully Verified" but its sources were not independently re-checked here; where it disagreed with earlier batting hand or bowling style data the earlier value was kept. The other 11 are 9 umpires and match officials plus 2 namesake ids that were cleared. Three namesake ids for "Mohammad Irfan" had been given one bio; two were cleared. 4 birth dates are 1 January (possible placeholders).
        - **Auction names**: 852 of 1,202 rows are matched to Cricsheet names. 78 near-matches were reviewed with team and season evidence; one impossible match (a 2017 draft pick matched to a player born in 2004) was rejected.
        """)

    with t5:
        ui.table(D.inventory())
        st.caption("Superseded files (earlier gap-file versions, the 10-year workbook) are not loaded.")
