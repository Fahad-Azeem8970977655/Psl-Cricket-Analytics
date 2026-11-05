import pandas as pd
import streamlit as st
from lib import data as D, ui


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    m = D.load_matches()
    m = m[m.season.isin(seasons)]
    t3, t4 = st.tabs(["Extremes", "Super overs & records"])

    with t3:
        c1, c2 = st.columns(2)
        inn = d.groupby(["match_idx", "innings", "bat_team"]).agg(runs=("total", "sum"), wk=("is_dismissal", "sum"), date=("date", "first")).reset_index()
        inn["date"] = inn.date.dt.date
        with c1:
            st.markdown("**Highest team totals**")
            ui.table(inn.sort_values("runs", ascending=False).head(10)[["date", "bat_team", "runs", "wk"]])
            st.markdown("**Lowest team totals (all out or full 20 overs)**")
            ui.table(inn.sort_values("runs").head(10)[["date", "bat_team", "runs", "wk"]])
        bi = D.batter_innings(d)
        bo = d.groupby(["match_idx", "bowler", "season"]).agg(wk=("bowler_wkt", "sum"), runs=("bowl_runs", "sum"), balls=("bowl_balls", "sum")).reset_index()
        with c2:
            st.markdown("**Highest individual scores**")
            ui.table(bi.sort_values("runs", ascending=False).head(10)[["season", "batter", "bat_team", "runs", "balls", "fours", "sixes"]])
            st.markdown("**Best bowling figures**")
            ui.table(bo.sort_values(["wk", "runs"], ascending=[False, True]).head(10)[["season", "bowler", "wk", "runs"]].round(0))
        mm = m[m.decided]
        st.markdown("**Closest finishes**")
        close = mm[((mm.margin_type == "runs") & (mm.margin_value <= 3)) | ((mm.margin_type == "wickets") & (mm.margin_value <= 1))]
        ui.table(close[["date", "stage", "team1", "team2", "winner", "margin_type", "margin_value"]].assign(date=lambda x: x.date.dt.date))
        st.markdown("**Biggest wins by runs**")
        ui.table(mm[mm.margin_type == "runs"].sort_values("margin_value", ascending=False).head(10)[["date", "team1", "team2", "winner", "margin_value"]].assign(date=lambda x: x.date.dt.date))

    with t4:
        so = m[m.result_type == "super over"]
        st.markdown("**Super-over matches** (their extra deliveries are excluded from all player statistics)")
        ui.table(so[["date", "stage", "team1", "team2", "winner"]].assign(date=lambda x: x.date.dt.date))
        fr = D.load_final_results()
        fr = fr[fr.season.isin(seasons)]
        tt = fr.champion.value_counts().rename_axis("team").reset_index(name="titles")
        fin = pd.concat([fr.champion, fr.runner_up]).value_counts().rename_axis("team").reset_index(name="finals played")
        st.markdown("**Titles and finals** (from the final results file)")
        ui.table(tt.merge(fin, on="team", how="outer").fillna(0).sort_values(["titles", "finals played"], ascending=False))
        st.caption("The workbook's notable-records table is not shown: its provenance and values could not be verified.")
