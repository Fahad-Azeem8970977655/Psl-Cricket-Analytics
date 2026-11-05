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


def _label(fig, x, y, text, size=13, color=INK, bg="rgba(255,255,255,0.86)", bold=False, yanchor="middle"):
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False, yanchor=yanchor,
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

    compact = height < 520                      # small ground (overview): fewer, smaller labels
    fs = 11 if compact else 13
    n = lambda v: f"{int(v):,}"
    run_share = 100 * sh(run_runs)

    # Sixes sit just outside the rope, fours in the outer band; both are coloured like their zone.
    _label(fig, 0, 1.33, f"Sixes {n(c['sixes'])} ({100 * sh(six_runs):.0f}% of runs)", size=fs, bold=True,
           color="#A8242B", bg="rgba(255,255,255,0.92)")
    _label(fig, 0, 0.70, f"Fours {n(c['fours'])} ({100 * sh(four_runs):.0f}%)", size=fs, bold=True,
           color="#7a5a14", bg="rgba(255,255,255,0.88)")

    # Everything run between the wickets is far too small to label inside the 30-yard circle,
    # so it is written under the ground and joined to the circle by a dotted line.
    pc = lambda x: f"{x:.0f}%" if x >= 1 else f"{x:.1f}%"
    dot_share = 100 * c["dots"] / max(c["balls"], 1)
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.41, y1=-1.30, line=dict(color="rgba(31,107,69,0.85)", width=1.5, dash="dot"))
    _label(fig, 0, -1.31,
           f"<b>Run between the wickets: {pc(run_share)} of runs</b><br>"
           f"Singles {n(c['singles'])} ({pc(100 * sh(c['singles']))} of runs)<br>"
           f"Twos {n(c['twos'])} ({pc(100 * sh(2 * c['twos']))})  \u00b7  Threes {n(c['threes'])} ({pc(100 * sh(3 * c['threes']))})<br>"
           f"Dots (no run) {n(c['dots'])} ({pc(dot_share)} of balls)",
           size=fs, bold=False, color="#164D32", bg="rgba(255,255,255,0.92)", yanchor="top")
    if not compact:
        _label(fig, 0.66, 0.02, "30-yard circle", size=10, bg="rgba(255,255,255,0)", color="#164D32")
        _label(fig, 0.80, -0.60, "boundary rope", size=10, bg="rgba(255,255,255,0.7)", color="#7a5a14")

    fig.update_xaxes(visible=False, range=[-1.45, 1.45])
    fig.update_yaxes(visible=False, range=[-2.25, 1.5], scaleanchor="x")
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=22, family=HEAD_FONT)), height=height,
                      margin=dict(l=0, r=0, t=46 if title else 6, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


FIELD_CAPTION = ("Zones show **how** runs were scored, not **where** the ball went: the data has no shot direction "
                 "(wagon-wheel) field. Inside the dotted 30-yard circle = runs run between the wickets (1s, 2s, 3s); "
                 "between the circle and the rope = fours; beyond the rope = sixes. Colour strength = share of the runs.")
