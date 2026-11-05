"""PSL Analytics 2016-2026 - entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="PSL Analytics 2016-2026", page_icon="🏏", layout="wide")

from lib import theme, ui  # noqa: E402

theme.apply()
ui.sidebar_filters()

pages = {
    "Overview": [
        st.Page("pages/p01_overview.py", title="Overview & data health", icon="🏏", default=True),
        st.Page("pages/p02_seasons_records.py", title="Seasons, champions & records", icon="🏆"),
    ],
    "Teams & matches": [
        st.Page("pages/p03_match_centre.py", title="Match centre", icon="📋"),
        st.Page("pages/p04_team_strength.py", title="Team strength (Elo)", icon="🛡️"),
        st.Page("pages/p05_toss_chase_h2h.py", title="Toss, chase & head-to-head", icon="🪙"),
        st.Page("pages/p06_venues_phases.py", title="Venues & phases", icon="🏟️"),
    ],
    "Players": [
        st.Page("pages/p07_batter_zones.py", title="Batter scoring zones (pitch)", icon="🎯"),
        st.Page("pages/p08_partnerships_order.py", title="Partnerships & batting order", icon="🤝"),
        st.Page("pages/p09_bowling_fielding.py", title="Bowling & fielding", icon="🌀"),
        st.Page("pages/p10_matchups.py", title="Batter vs bowler", icon="⚔️"),
        st.Page("pages/p11_impact.py", title="Player impact ratings", icon="📊"),
        st.Page("pages/p12_clusters_anomalies.py", title="Clusters, similar players & anomalies", icon="🧬"),
    ],
    "Machine learning & honesty": [
        st.Page("pages/p13_win_prediction.py", title="Win prediction", icon="🔮"),
        st.Page("pages/p14_luck_skill.py", title="Luck vs skill", icon="🎲"),
    ],
    "Franchise": [
        st.Page("pages/p15_auction_squads.py", title="Auction, squads & value for money", icon="💰"),
    ],
}
st.navigation(pages).run()
