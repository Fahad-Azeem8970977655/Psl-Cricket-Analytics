"""Plot helpers, including the cricket ground / pitch graphic used for scoring areas."""
import numpy as np
import plotly.graph_objects as go

PITCH_TAN = "#d8c9a0"
INK = "#17231D"
ZONE = {"running": (31, 107, 69), "four": (201, 150, 46), "six": (168, 36, 43)}   # turf, brass, leather
HEAD_FONT = "Barlow Condensed, Arial Narrow, sans-serif"


def _rgba(rgb, strength, lo=0.16, hi=0.92):
    a = lo + (hi - lo) * max(0.0, min(1.0, strength))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f})"


def _circle(fig, r, fill, line="rgba(255,255,255,0.9)", width=2, dash=None):
    fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, fillcolor=fill,
                  line=dict(color=line, width=width, dash=dash), layer="below")


def _stripes(fig, radius=1.22, bands=14, alpha=0.055):
    """Mown-grass bands clipped to the ground circle (purely visual)."""
    edges = np.linspace(-radius, radius, bands + 1)
    for i in range(0, bands, 2):
        x = np.linspace(edges[i], edges[i + 1], 12)
        y = np.sqrt(np.maximum(radius ** 2 - x ** 2, 0))
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[y, -y[::-1]], mode="none", fill="toself",
                                 fillcolor=f"rgba(0,0,0,{alpha})", hoverinfo="skip", showlegend=False))


def _label(fig, x, y, text, size=13, color=INK, bg="rgba(255,255,255,0.86)", bold=False):
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False,
                       font=dict(size=size, color=color), bgcolor=bg, borderpad=3, align="center")


def field_figure(c: dict, title: str = "", height: int = 640) -> go.Figure:
    """Ground + 22-yard pitch coloured by *how* runs were scored.

    The dataset has no shot direction, so zones are by outcome, not by angle:
    inside the 30-yard circle = runs run between the wickets (1s, 2s, 3s),
    between circle and rope = fours, beyond the rope = sixes.
    `c` needs: runs, balls, dots, singles, twos, threes, fours, sixes.
    """
    run_runs = c["singles"] + 2 * c["twos"] + 3 * c["threes"]
    four_runs, six_runs = 4 * c["fours"], 6 * c["sixes"]
    tot = max(c["runs"], 1)
    sh = lambda x: x / tot

    fig = go.Figure()
    _circle(fig, 1.22, _rgba(ZONE["six"], sh(six_runs) * 2.2), line="rgba(0,0,0,0.0)")
    _circle(fig, 1.0, _rgba(ZONE["four"], sh(four_runs) * 1.8), line="white", width=4)
    _circle(fig, 0.40, _rgba(ZONE["running"], sh(run_runs) * 1.4), line="white", width=2, dash="dot")
    _stripes(fig)
    # pitch strip, creases, stumps
    fig.add_shape(type="rect", x0=-0.05, x1=0.05, y0=-0.19, y1=0.19, fillcolor=PITCH_TAN,
                  line=dict(color="rgba(80,60,30,0.7)", width=1), layer="below")
    for y in (-0.13, 0.13):
        fig.add_shape(type="line", x0=-0.085, x1=0.085, y0=y, y1=y, line=dict(color="white", width=2))
    for y in (-0.19, 0.19):
        for x in (-0.018, 0, 0.018):
            fig.add_shape(type="line", x0=x, x1=x, y0=y - 0.012, y1=y + 0.012, line=dict(color="#3b2f1c", width=2))

    _label(fig, 0, 1.15, f"Sixes {c['sixes']}  ({six_runs} runs, {100 * sh(six_runs):.0f}%)", bold=True)
    _label(fig, 0, 0.72, f"Fours {c['fours']}  ({four_runs} runs, {100 * sh(four_runs):.0f}%)", bold=True)
    _label(fig, -0.22, 0.04, f"Singles<br>{c['singles']}", size=13, bold=True)
    _label(fig, 0.23, 0.04, f"Twos {c['twos']}<br>Threes {c['threes']}", size=12, bold=True)
    _label(fig, 0, -0.01, f"Dots<br>{c['dots']}", size=10, bg="rgba(255,255,255,0.0)", color="#3b2f1c")
    _label(fig, 0, 0.30, "bowler's end", size=10, bg="rgba(255,255,255,0)", color="#3b2f1c")
    _label(fig, 0, -0.30, "batter's end", size=10, bg="rgba(255,255,255,0)", color="#3b2f1c")
    _label(fig, 0.64, 0.0, "30-yard circle", size=10, bg="rgba(255,255,255,0)", color="#164D32")
    _label(fig, 0.80, -0.60, "boundary rope", size=10, bg="rgba(255,255,255,0.7)", color="#7a5a14")

    fig.update_xaxes(visible=False, range=[-1.3, 1.3])
    fig.update_yaxes(visible=False, range=[-1.3, 1.3], scaleanchor="x")
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=22, family=HEAD_FONT)), height=height,
                      margin=dict(l=0, r=0, t=46 if title else 6, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


FIELD_CAPTION = ("Zones show **how** runs were scored, not **where** the ball went: the data has no shot direction "
                 "(wagon-wheel) field. Inside the dotted 30-yard circle = runs run between the wickets (1s, 2s, 3s); "
                 "between the circle and the rope = fours; beyond the rope = sixes. Colour strength = share of the runs.")
