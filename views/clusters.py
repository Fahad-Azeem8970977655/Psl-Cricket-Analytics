import plotly.express as px
import streamlit as st
from lib import data as D, ml, ui


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    kind = st.radio("Players", ["Batters", "Bowlers"], horizontal=True)
    if kind == "Batters":
        minb = st.slider("Minimum balls faced", 60, 800, 150, 10)
        prof, feats = ml.batter_profiles(d, minb), ml.BAT_FEATS
    else:
        minb = st.slider("Minimum balls bowled", 60, 800, 120, 10)
        prof, feats = ml.bowler_profiles(d, minb), ml.BOWL_FEATS
    if len(prof) < 12:
        st.warning("Too few players with these filters. Lower the minimum or add seasons.")
        st.stop()
    prof = prof.reset_index(drop=True)

    sil = ml.silhouette_scan(prof, feats, range(2, min(8, len(prof) // 3 + 1)))
    best = int(sil.loc[sil.silhouette.idxmax(), "k"])
    k = st.slider("Number of clusters", 2, 7, min(max(best, 3), 7), help=f"Highest silhouette score is at k={best}")
    cl, z = ml.run_cluster(prof, feats, int(k))
    names = ml.label_clusters(z)
    cl["style"] = cl.cluster.map(names)

    c1, c2 = st.columns([3, 1])
    with c1:
        ui.show(px.scatter(cl, x="pc1", y="pc2", color="style", hover_name="player", hover_data={f: ":.1f" for f in feats[:4]} | {"pc1": False, "pc2": False},
                           title="Player styles (PCA map)"))
    with c2:
        ui.show(px.line(sil, x="k", y="silhouette", markers=True, title="Silhouette by k"))
        st.caption("Scores near 0.2-0.3 mean the groups overlap: play styles form a continuum more than separate boxes.")

    st.markdown("**What defines each cluster** (average standardised value)")
    ui.show(px.imshow(z.rename(index=names).round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2))
    ui.table(cl.sort_values(["cluster", feats[0]], ascending=[True, False])[["style", "player"] + feats].round(1))

    st.markdown("### Find similar players")
    who = st.selectbox("Player", cl.player.sort_values().tolist())
    sim = ml.nearest_players(cl, feats, who, 8)
    ui.table(sim[["player", "similarity", "style"] + feats[:5]].round(2))
    st.caption("Similarity is cosine similarity of the standardised profile, so it matches style, not quality.")

