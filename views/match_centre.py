import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import cricket as C, data as D, ml, theme, ui


def _label(r):
    res = r.winner if isinstance(r.winner, str) else "no result"
    return f"{r.date:%Y-%m-%d}  {r.team1} v {r.team2}  [{r.stage}]  - {res}"


def _result_line(r, sc):
    if r.result_type == "no result":
        return "No result"
    if r.result_type == "super over":
        return f"Scores tied. {r.winner} won the super over"
    if r.margin_type == "runs":
        return f"{r.winner} won by {int(r.margin_value)} runs"
    if r.margin_type == "wickets":
        return f"{r.winner} won by {int(r.margin_value)} wickets"
    return f"{r.winner} won" if isinstance(r.winner, str) else "-"


def render():
    seasons = ui.selected_seasons()
    m = D.load_matches()
    m = m[m.season.isin(seasons)].sort_values("date", ascending=False)
    c1, c2 = st.columns(2)
    team = c1.selectbox("Team", ["All"] + D.TEAMS)
    stage = c2.selectbox("Stage", ["All", "league", "qualifier", "eliminator", "final"])
    if team != "All":
        m = m[(m.team1 == team) | (m.team2 == team)]
    if stage != "All":
        m = m[m.stage == stage]
    if m.empty:
        st.warning("No matches with these filters.")
        st.stop()
    labels = {int(r.match_idx): _label(r) for r in m.itertuples()}
    idx = st.selectbox("Match", list(labels), format_func=lambda i: labels[i])
    r = m[m.match_idx == idx].iloc[0]
    sc = C.scorecard(int(idx))

    sides = []
    for inn in (1, 2):
        if inn in sc:
            s = sc[inn]
            sides.append((s["team"], f"{s['total']['runs']}/{s['total']['wkts']}", f"{s['total']['overs']} overs, run rate {s['total']['run_rate']}",
                          D.TEAM_COLORS.get(s["team"], "#1F6B45")))
    where = f"{r.venue}, {r.city}" if isinstance(r.city, str) else r.venue
    number = f"Match {int(r.match_number)}" if pd.notna(r.match_number) else r.stage.capitalize()
    toss = f"{r.toss_winner} won the toss and chose to {r.toss_decision}" if isinstance(r.toss_winner, str) else ""
    theme.scoreboard(sides, _result_line(r, sc), [f"{r.date:%A %d %B %Y}", where, f"{number}, season {int(r.season)}", toss])

    tabs = ["Scorecard", "Worm & Manhattan", "Partnerships", "Phases", "Win probability", "Match info"]
    so = C.super_over(int(idx)) if r.result_type == "super over" else None
    if so is not None:
        tabs.append("Super over")
    tt = st.tabs(tabs)

    with tt[0]:
        for inn in (1, 2):
            if inn not in sc:
                continue
            s = sc[inn]
            st.markdown(f"#### {s['team']} batting: {s['total']['runs']}/{s['total']['wkts']} ({s['total']['overs']} overs)")
            ui.table(s["batting"])
            e = s["extras"]
            st.caption(f"Extras {e['total']} (b {e['byes']}, lb {e['leg_byes']}, w {e['wides']}, nb {e['no_balls']}, pen {e['penalty']})"
                       + (f" | Did not bat: {', '.join(s['did_not_bat'])}" if s["did_not_bat"] else ""))
            if len(s["fow"]):
                st.caption("Fall of wickets: " + "; ".join(f"{int(x.wicket)}-{x.score} ({x.batter}, {x.over} ov)" for x in s["fow"].itertuples()))
            st.markdown(f"**{s['opp']} bowling**")
            ui.table(s["bowling"])

    with tt[1]:
        fig = go.Figure()
        for inn in (1, 2):
            if inn not in sc:
                continue
            s, w = sc[inn], sc[inn]["worm"]
            col = D.TEAM_COLORS.get(s["team"], "#1F6B45")
            fig.add_trace(go.Scatter(x=w.ball / 6, y=w.runs, mode="lines", name=s["team"], line=dict(color=col, width=3)))
            k = w[w.wicket]
            fig.add_trace(go.Scatter(x=k.ball / 6, y=k.runs, mode="markers", name=f"{s['team']} wicket", marker=dict(color=col, size=9, symbol="x"), hovertext=k.who))
        fig.update_layout(title="Worm: runs against overs (x = wicket)", xaxis_title="overs", yaxis_title="runs", height=420)
        ui.show(fig)
        fig = go.Figure()
        for inn in (1, 2):
            if inn not in sc:
                continue
            s, o = sc[inn], sc[inn]["by_over"]
            col = D.TEAM_COLORS.get(s["team"], "#1F6B45")
            fig.add_trace(go.Bar(x=o.over, y=o.runs, name=s["team"], marker_color=col))
            k = o[o.wkts > 0]
            fig.add_trace(go.Scatter(x=k.over, y=k.runs + 0.6, mode="text", text=["W" * int(n) for n in k.wkts], showlegend=False,
                                     textfont=dict(color=col, size=14)))
        fig.update_layout(title="Manhattan: runs in each over (W = wicket)", barmode="group", xaxis_title="over", yaxis_title="runs", height=400)
        ui.show(fig)

    with tt[2]:
        pt = C.partnership_table()
        pp = pt[pt.match_idx == idx].sort_values(["innings", "wicket_no"])
        for inn in (1, 2):
            q = pp[pp.innings == inn]
            if q.empty:
                continue
            st.markdown(f"**{q.bat_team.iloc[0]}**")
            lab = q.wicket_no.astype(int).astype(str) + " " + q.pair
            fig = go.Figure()
            fig.add_trace(go.Bar(y=lab, x=q.runs_lo, name="first batter (A-Z)", orientation="h", marker_color="#1F6B45"))
            fig.add_trace(go.Bar(y=lab, x=q.runs_hi, name="second batter", orientation="h", marker_color="#C9962E"))
            fig.add_trace(go.Bar(y=lab, x=q.extras, name="extras", orientation="h", marker_color="#9aa5a0"))
            fig.update_layout(barmode="stack", height=60 + 42 * len(q), yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0), xaxis_title="runs")
            ui.show(fig, key=f"pp_{inn}")
            ui.table(q[["wicket_no", "pair", "runs", "balls", "runs_lo", "runs_hi", "extras", "ended"]])

    with tt[3]:
        rows = []
        for inn in (1, 2):
            if inn in sc:
                ph = sc[inn]["phases"].assign(team=sc[inn]["team"])
                rows.append(ph)
        if rows:
            ph = pd.concat(rows)
            ui.show(px.bar(ph, x="phase", y="runs", color="team", barmode="group", text="wkts", category_orders={"phase": D.PHASES},
                           color_discrete_map=D.TEAM_COLORS, title="Runs by phase (labels = wickets lost)"))
            ui.table(ph[["team", "phase", "runs", "wkts", "balls", "run_rate"]])

    with tt[4]:
        if r.result_type in ("normal", "super over") and 2 in sc:
            if st.checkbox("Show the chase win-probability curve (trains the model on first use, about 20 seconds)", value=False):
                c = ml.chase_wpa()
                c = c[c.match_idx == idx].reset_index(drop=True)
                c["delivery"] = np.arange(1, len(c) + 1)
                chaser = sc[2]["team"]
                fig = go.Figure(go.Scatter(x=c.delivery, y=100 * c.wp, mode="lines", name=f"{chaser} win %", line=dict(color=D.TEAM_COLORS.get(chaser, "#1F6B45"), width=3)))
                w = c[c.is_dismissal]
                fig.add_trace(go.Scatter(x=w.delivery, y=100 * w.wp, mode="markers", name="wicket", marker=dict(color="#A8242B", size=9)))
                fig.add_hline(y=50, line_dash="dot")
                fig.update_layout(title=f"Chasing win probability: {chaser}", yaxis_title="win probability %", xaxis_title="delivery of the chase", height=380)
                ui.show(fig)
                st.caption("Model of the match state (runs, balls, wickets), not of team quality. See the Win prediction page for how it is validated.")
        else:
            st.info("Win probability is shown for completed chases.")

    with tt[5]:
        ui.kpis([("Toss won by", r.toss_winner), ("Chose to", r.toss_decision), ("Player of the match", r.player_of_match if isinstance(r.player_of_match, str) else "-")])
        st.caption(f"Umpires: {r.umpire_1}, {r.umpire_2}" + (f" | TV umpire: {r.tv_umpire}" if isinstance(r.tv_umpire, str) else "")
                   + (f" | Referee: {r.match_referee}" if isinstance(r.match_referee, str) else ""))
        xi = D.load_xi()
        xi = xi[xi.match_idx == idx]
        c1, c2 = st.columns(2)
        for col, t in zip((c1, c2), (r.team1, r.team2)):
            with col:
                st.markdown(f"**{t}: players listed**")
                ui.table(xi[xi.team == t][["player"]])

    if so is not None:
        with tt[6]:
            st.markdown("**Super over deliveries** (not counted in any player statistic)")
            ui.table(so)
