import pandas as pd
import plotly.express as px
import streamlit as st
from lib import data as D, ui, viz


def render():
    seasons = ui.selected_seasons()
    d = D.load_deliveries()
    d = d[d.season.isin(seasons)]
    bs = D.batting_summary(d)
    eligible = bs[bs.balls >= 60].sort_values("runs", ascending=False).batter.tolist()

    mode = st.radio("Show", ["Player", "Team", "Whole league"], horizontal=True)
    f1, f2, f3 = st.columns(3)
    phases = f1.multiselect("Phase", D.PHASES, default=D.PHASES)
    btype = f2.selectbox("Bowler type", ["All", "Pace", "Spin"])
    inn = f3.selectbox("Innings", ["Both", "Batting first", "Chasing"])

    base = d[d.phase.isin(phases or D.PHASES)]
    if btype != "All":
        base = base[base.bowl_type == btype]
    if inn != "Both":
        base = base[base.innings == (1 if inn == "Batting first" else 2)]

    subsets = []
    if mode == "Player":
        a = st.selectbox("Batter", eligible)
        b = st.selectbox("Compare with (optional)", ["(none)"] + [p for p in eligible if p != a])
        subsets.append((a, base[base.batter == a]))
        if b != "(none)":
            subsets.append((b, base[base.batter == b]))
        if btype != "All":
            cov = (d[d.batter == a].bowl_type != "Unknown").mean()
            st.caption(f"Bowler-type data is only known for players in the bio file: {100 * cov:.0f}% of {a}'s balls are covered.")
    elif mode == "Team":
        t = st.selectbox("Team", D.TEAMS)
        subsets.append((t, base[base.bat_team == t]))
    else:
        subsets.append(("PSL (all batters)", base))

    if not any(len(s) for _, s in subsets):
        st.warning("No deliveries match these filters.")
        st.stop()

    cols = st.columns(len(subsets))
    for col, (name, df) in zip(cols, subsets):
        with col:
            if len(df) == 0:
                st.warning(f"No deliveries for {name} with these filters.")
                continue
            c = D.run_mix(df)
            s = D.batting_summary(df.assign(_k=name), "_k").iloc[0]
            ui.kpis([("Runs", c["runs"]), ("Balls", c["balls"]), ("Strike rate", f"{s.sr:.1f}")])
            ui.kpis([("Fours", c["fours"]), ("Sixes", c["sixes"]), ("Singles", c["singles"])])
            ui.kpis([("Dot %", f"{s.dot_pct:.0f}%"), ("Boundary %", f"{s.boundary_pct:.0f}%"), ("Runs in boundaries", f"{s.boundary_runs_pct:.0f}%")])
            ui.show(viz.field_figure(c, title=name), key=f"field_{name}")
            runs_by = {"Dots": 0, "Singles": c["singles"], "Twos": 2 * c["twos"], "Threes": 3 * c["threes"],
                       "Fours": 4 * c["fours"], "Sixes": 6 * c["sixes"]}
            balls_by = {"Dots": c["dots"], "Singles": c["singles"], "Twos": c["twos"], "Threes": c["threes"],
                        "Fours": c["fours"], "Sixes": c["sixes"]}
            mix = pd.DataFrame({"Outcome": list(balls_by), "Balls": list(balls_by.values()), "Runs": list(runs_by.values())})
            mix["% of balls"] = 100 * mix.Balls / max(mix.Balls.sum(), 1)
            mix["% of runs"] = 100 * mix.Runs / max(c["runs"], 1)
            st.markdown("**Where the balls and the runs go**")
            ui.table(mix.assign(**{"% of balls": mix["% of balls"].round(1), "% of runs": mix["% of runs"].round(1)}))
            long = mix.melt("Outcome", value_vars=["% of balls", "% of runs"], var_name="Share", value_name="pct")
            fig = px.bar(long, x="Outcome", y="pct", color="Share", barmode="group", text=long.pct.round(1),
                         title="Share of balls vs share of runs", color_discrete_sequence=["#8A8F98", "#1F6B45"])
            fig.update_yaxes(title="%")
            ui.show(fig, key=f"mix_{name}")
    st.caption(viz.FIELD_CAPTION)

    name0, df0 = subsets[0]
    st.markdown(f"### More on {name0}")
    t1, t2, t3, t4 = st.tabs(["By over", "By season", "Innings", "Against the league"])
    with t1:
        o = df0.groupby("over").agg(runs=("runs_batter", "sum"), balls=("bat_balls", "sum"), b=("is_four", "sum"), s=("is_six", "sum")).reset_index()
        o["strike_rate"] = 100 * o.runs / o.balls
        o["boundaries"] = o.b + o.s
        ui.show(px.bar(o, x="over", y="runs", color="strike_rate", color_continuous_scale="YlOrRd", title="Runs by over (colour = strike rate)"))
    with t2:
        x = D.batting_summary(df0, "season")
        c1, c2 = st.columns(2)
        with c1:
            ui.show(px.bar(x, x="season", y="runs", text="runs", title="Runs per season"))
        with c2:
            ui.show(px.line(x, x="season", y="sr", markers=True, title="Strike rate per season"))
    with t3:
        if mode == "Player":
            inn_t = D.batter_innings(df0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Fifties", int(((inn_t.runs >= 50) & (inn_t.runs < 100)).sum()))
            c2.metric("Hundreds", int((inn_t.runs >= 100).sum()))
            c3.metric("Highest score", int(inn_t.runs.max()))
            ui.show(px.histogram(inn_t, x="runs", nbins=20, title="Distribution of innings scores"))
            st.caption("Innings counts follow the filters above (a phase or bowler-type filter splits an innings).")
        else:
            st.info("Innings distribution is shown for individual batters.")
    with t4:
        ref = D.batting_summary(base.assign(_k="league"), "_k").iloc[0]
        me = D.batting_summary(df0.assign(_k="x"), "_k").iloc[0]
        cmp_ = pd.DataFrame({"measure": ["Strike rate", "Boundary %", "Dot %", "Balls per boundary"],
                             name0: [me.sr, me.boundary_pct, me.dot_pct, me.balls_per_boundary],
                             "League": [ref.sr, ref.boundary_pct, ref.dot_pct, ref.balls_per_boundary]})
        ui.show(px.bar(cmp_.melt("measure", var_name="who", value_name="value"), x="measure", y="value", color="who", barmode="group",
                       title="Against the league average (same filters)", color_discrete_sequence=["#1F6B45", "#C9962E"]))
