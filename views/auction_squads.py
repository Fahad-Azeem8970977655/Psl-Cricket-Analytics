import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from lib import data as D, ui


def render():
    seasons = ui.selected_seasons()
    ad, cat, ts = D.load_auction(), D.load_category_prices(), D.load_team_seasons()
    bio, sq = D.load_bio(), D.load_squads()
    ad, ts = ad[ad.season.isin(seasons)], ts[ts.season.isin(seasons)]

    t1, t2, t3, t4 = st.tabs(["Draft & auction", "Salary brackets", "Squad age & mix", "Captains, coaches, owners"])

    with t1:
        cnt = ad.dropna(subset=["draft_category"]).groupby(["season", "draft_category"]).size().reset_index(name="players")
        ui.show(px.bar(cnt, x="season", y="players", color="draft_category", title="Draft picks by category"))
        st.caption("2018 includes a 'Supplementary' category. 2026 replaced the draft with an auction, so it has no draft categories.")
        a26 = ad[(ad.season == 2026) & (ad.price_type == "auction")]
        if len(a26):
            sp = a26.groupby("team").final_price_pkr_cr.agg(["sum", "count"]).reset_index().rename(columns={"sum": "spent_crore", "count": "players_bought"})
            c1, c2 = st.columns(2)
            with c1:
                ui.show(px.bar(sp.sort_values("spent_crore"), x="spent_crore", y="team", orientation="h", color="team", color_discrete_map=D.TEAM_COLORS,
                               title="2026 auction spend by team (PKR crore)"))
            with c2:
                fig = px.scatter(a26, x="base_price_pkr_cr", y="final_price_pkr_cr", hover_name="player", color="team", color_discrete_map=D.TEAM_COLORS,
                                 title="2026: base price vs price paid")
                fig.add_shape(type="line", x0=0, y0=0, x1=a26.final_price_pkr_cr.max(), y1=a26.final_price_pkr_cr.max(), line=dict(dash="dot"))
                ui.show(fig)
            ui.table(a26.sort_values("final_price_pkr_cr", ascending=False).head(15)[["player", "team", "base_price_pkr_cr", "final_price_pkr_cr"]])
            st.caption("2026 values are unverified here (they post-date the checks that could be run).")
        nat = ad.dropna(subset=["nationality"])
        if len(nat):
            nat = nat.assign(group=np.where(nat.nationality == "Pakistan", "Pakistan", "Overseas")).groupby(["season", "group"]).size().reset_index(name="players")
            ui.show(px.bar(nat, x="season", y="players", color="group", barmode="relative", title=f"Nationality of drafted players (known for {100 * ad.nationality.notna().mean():.0f}% of rows)"))

    with t2:
        mid = cat.assign(mid=cat[["usd_low", "usd_high"]].mean(axis=1))
        fig = px.line(mid.dropna(subset=["mid"]), x="season", y="mid", color="category", markers=True,
                      category_orders={"category": ["Platinum", "Diamond", "Gold", "Silver", "Emerging"]}, title="Category salary (USD, midpoint of the stated range)")
        ui.show(fig)
        ui.table(cat[["season", "category", "price", "currency", "notes"]])
        st.caption("Mixed sources: 2020 is a proposed range, 2024 and 2025 come from secondary sports sites. 2020 Silver/Emerging and all of 2026 are blank. "
                   "Amounts are in USD as quoted; no exchange-rate conversion was applied. The workbook's PKR price sheet (S5) is not used because it does not agree with these figures.")

    with t3:
        s2 = sq.merge(bio[["player", "dob", "nationality", "role"]], on="player", how="left", suffixes=("", "_bio"))
        s2["age"] = (pd.to_datetime(s2.season.astype(str) + "-06-01") - s2.dob).dt.days / 365.25
        cov = s2.dob.notna().mean()
        st.warning("Birth dates and nationalities come from a supplied bio workbook marked 'Fully Verified' whose sources were not independently re-checked here; "
                   "a few players (for example namesakes) may still be wrong.")
        st.caption(f"Birth dates are known for {100 * cov:.0f}% of squad entries (players in the bio file, mostly regulars). Squads are the players who appeared in matches.")
        age = s2.dropna(subset=["age"]).groupby(["team", "season"]).age.mean().reset_index()
        ui.show(px.imshow(age.pivot(index="team", columns="season", values="age").round(1), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                          title="Average age of squad members with known birth dates"))
        d = D.load_deliveries()
        d = d[d.season.isin(seasons)]
        key = d.assign(_k=d.season.astype(str) + "|" + d.batter)
        ps = D.batting_summary(key, "_k")
        ps[["season", "player"]] = ps._k.str.split("|", n=1, expand=True)
        ps["season"] = ps.season.astype(int)
        ps = ps[ps.balls >= 100].merge(bio[["player", "dob"]], on="player", how="left").dropna(subset=["dob"])
        ps["age"] = (pd.to_datetime(ps.season.astype(str) + "-06-01") - ps.dob).dt.days / 365.25
        if len(ps) > 10:
            ps["age band"] = pd.cut(ps.age, [15, 22, 26, 30, 34, 50], labels=["under 22", "22-25", "26-29", "30-33", "34+"])
            ui.show(px.box(ps, x="age band", y="sr", points="all", hover_name="player", title="Batting strike rate by age (player-seasons with 100+ balls)"))
            st.caption("Descriptive only: survivor effects and role differences mean this is not an age curve.")
        s2["overseas"] = s2.nationality.notna() & (s2.nationality != "Pakistan")
        ov = s2.dropna(subset=["nationality"]).groupby("season").overseas.mean().reset_index()
        ui.show(px.line(ov, x="season", y="overseas", markers=True, title="Overseas share of squad members with known nationality"))

    with t4:
        team = st.selectbox("Team", D.TEAMS)
        x = ts[ts.team == team].sort_values("season")
        ui.table(x[["season", "captain", "head_coach", "owner_franchise", "nominal_home_ground"]])
        cap = ts.groupby(["team", "captain"]).season.agg(["count", "min", "max"]).reset_index().rename(columns={"count": "seasons"})
        ui.show(px.bar(cap[cap.team == team], x="captain", y="seasons", title=f"{team}: seasons as captain"))
        fr = D.load_final_results()
        ch = fr.merge(ts, left_on=["season", "champion"], right_on=["season", "team"], how="left")
        st.markdown("**Captains and coaches of title-winning teams**")
        ui.table(ch[["season", "champion", "captain", "head_coach"]])
        st.caption("The ground column is the franchise's nominal home ground, not where matches were played. 2026 values are unverified.")
