import streamlit as st

from lib import data as D, theme, ui, viz
from views import overview

m = D.load_matches()
d = D.load_deliveries()
mix = D.run_mix(d)
share = lambda n: 100 * n / max(mix["runs"], 1)
running = mix["singles"] + 2 * mix["twos"] + 3 * mix["threes"]

left, right = st.columns([5, 6], gap="large")
with left:
    theme.hero_text(
        "Every ball of eleven PSL seasons",
        [f"<b>{len(d):,} deliveries</b> from <b>{len(m)} matches</b>, ball by ball, from the first season in 2016 to the 2026 final.",
         f"Across all of them, <b>{share(running):.0f}%</b> of bat runs were ones, twos and threes, <b>{share(4 * mix['fours']):.0f}%</b> came from fours "
         f"and <b>{share(6 * mix['sixes']):.0f}%</b> from sixes.",
         "The ground on the right shows that split. Open the batter scoring zones page to draw it for one player."],
        "Statistics exclude super overs. The 2026 season is not fully verified.")
with right:
    ui.show(viz.field_figure(mix, title="", height=470), key="overview_field")

st.write("")
overview.render()
