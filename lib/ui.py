"""Shared UI helpers: sidebar filters, headers, KPI rows."""
import textwrap

import streamlit as st

from . import data as D
from . import theme


def sidebar_filters():
    m = D.load_matches()
    seasons = sorted(int(x) for x in m.season.unique())
    with st.sidebar:
        st.markdown("#### Seasons")
        st.multiselect("Seasons", seasons, default=seasons, key="seasons_sel", label_visibility="collapsed")
        st.caption("Applies to pages that use match or ball-by-ball data. 2026 data is not fully verified.")


def selected_seasons():
    m = D.load_matches()
    allx = sorted(int(s) for s in m.season.unique())
    sel = st.session_state.get("seasons_sel") or allx
    return [s for s in sel if s in allx] or allx


def header(title: str, subtitle: str = ""):
    theme.page_header(title, subtitle)


def kpis(items):
    """items: list of (label, value[, tooltip]) shown as one scoreboard strip."""
    theme.strip(items)


def fmt_pct(x, d=1):
    return "-" if x != x else f"{x:.{d}f}%"


# ---- version-proof wrappers (Streamlit renamed use_container_width -> width="stretch")
import inspect


def _has(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


_PLOT_NEW = _has(st.plotly_chart, "width")
_DF_NEW = _has(st.dataframe, "width")


def _nice(t):
    return t.replace("_", " ") if isinstance(t, str) else t


def _hide_redundant_legend(fig):
    """A bar chart coloured by the same field as its axis (one bar per trace) does not need a legend."""
    try:
        traces = [t for t in fig.data if t.showlegend is not False]
        if not traces:
            return
        for t in traces:
            if t.type != "bar" or t.name is None:
                return
            xs = list(t.x) if t.x is not None else []
            ys = list(t.y) if t.y is not None else []
            if not ((len(xs) == 1 and xs[0] == t.name) or (len(ys) == 1 and ys[0] == t.name)):
                return
        fig.update_layout(showlegend=False)
    except Exception:
        pass


def _tidy(fig):
    """Readable axis / legend titles (no raw column names such as margin_value)."""
    _hide_redundant_legend(fig)
    try:
        lay = fig.layout
        lay.legend.title.text = _nice(lay.legend.title.text)
        for ax in (lay.xaxis, lay.yaxis):
            if ax.title.text:
                ax.title.text = _nice(ax.title.text)
        cb = lay.coloraxis.colorbar
        if cb.title.text:
            cb.title.text = _nice(cb.title.text)
    except Exception:
        pass
    return fig


def show(fig, key=None):
    fig = _tidy(fig)
    if _PLOT_NEW:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, **kw):
    if _DF_NEW:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def md(text: str):
    st.markdown(textwrap.dedent(text))


def views(options: dict, key: str):
    """Show one of several views (only the chosen one runs, which keeps heavy models lazy)."""
    names = list(options)
    if len(names) == 1:
        options[names[0]]()
        return
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("View", names, default=names[0], key=key, label_visibility="collapsed") or names[0]
    else:
        choice = st.radio("View", names, horizontal=True, key=key, label_visibility="collapsed")
    st.write("")
    options[choice]()
