"""Shared Streamlit UI: styling, cached data access, maps and charts."""
from __future__ import annotations

import html
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from . import aqi as A
from . import config, engine, geo
from .data import citizen, live

# ----------------------------------------------------------------------------
# Design tokens (validated categorical palette + chart chrome)
# ----------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SOURCE_COLORS = {"industrial": SERIES[0], "burning": SERIES[1], "traffic": SERIES[2], "dust": SERIES[3]}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def hex_rgb(h: str, a: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


CSS = """
<style>
:root { --vs-ink:#0b0b0b; --vs-ink2:#52514e; --vs-muted:#898781; --vs-line:#e1e0d9; --vs-card:#ffffff; --vs-accent:#0e7490; }
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }
.vs-hero { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  border-bottom:1px solid var(--vs-line); padding-bottom:.7rem; margin-bottom:1rem; }
.vs-hero h1 { font-size:1.65rem; line-height:1.2; margin:0; color:var(--vs-ink); font-weight:700; letter-spacing:-.01em; }
.vs-hero p { margin:.25rem 0 0 0; color:var(--vs-ink2); font-size:.95rem; }
.vs-chips { display:flex; gap:.4rem; flex-wrap:wrap; }
.vs-chip { display:inline-flex; align-items:center; gap:.35rem; border:1px solid var(--vs-line); background:#fff;
  border-radius:999px; padding:.18rem .6rem; font-size:.74rem; color:var(--vs-ink2); white-space:nowrap; }
.vs-dot { width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
.vs-kpi { background:var(--vs-card); border:1px solid var(--vs-line); border-radius:12px; padding:.8rem .95rem; height:100%; }
.vs-kpi .l { font-size:.78rem; color:var(--vs-ink2); margin-bottom:.15rem; }
.vs-kpi .v { font-size:1.7rem; font-weight:650; color:var(--vs-ink); line-height:1.15; }
.vs-kpi .s { font-size:.76rem; color:var(--vs-muted); margin-top:.2rem; }
.vs-card { background:var(--vs-card); border:1px solid var(--vs-line); border-radius:12px; padding:.85rem 1rem; margin-bottom:.7rem; }
.vs-card h4 { margin:.1rem 0 .35rem 0; font-size:1rem; color:var(--vs-ink); }
.vs-card .meta { color:var(--vs-ink2); font-size:.8rem; }
.vs-card ul { margin:.35rem 0 0 1.1rem; padding:0; }
.vs-card li { font-size:.84rem; color:var(--vs-ink2); margin-bottom:.12rem; }
.vs-aqi { display:inline-block; border-radius:6px; padding:.08rem .45rem; font-size:.78rem; font-weight:600; }
.vs-level { display:inline-block; border-radius:6px; padding:.08rem .45rem; font-size:.74rem; font-weight:700; color:#fff; letter-spacing:.02em; }
.vs-bar { height:.45rem; background:#eef0f3; border-radius:999px; overflow:hidden; margin:.2rem 0 .1rem 0; }
.vs-bar > span { display:block; height:100%; border-radius:999px; }
.vs-row { display:flex; justify-content:space-between; gap:.6rem; font-size:.8rem; color:var(--vs-ink2); }
.vs-section { font-size:1.05rem; font-weight:650; color:var(--vs-ink); margin:.9rem 0 .4rem 0; }
.vs-note { font-size:.8rem; color:var(--vs-muted); }
.vs-legend { display:flex; gap:.9rem; flex-wrap:wrap; font-size:.75rem; color:var(--vs-ink2); margin:.3rem 0 .2rem 0; }
.vs-legend span { display:inline-flex; align-items:center; gap:.3rem; }
.vs-sw { width:.8rem; height:.8rem; border-radius:3px; display:inline-block; }
[data-testid="stSidebar"] .vs-brand { font-weight:750; font-size:1.25rem; color:var(--vs-ink); margin-bottom:0; }
[data-testid="stSidebar"] .vs-tag { color:var(--vs-ink2); font-size:.82rem; margin-top:.1rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


# ----------------------------------------------------------------------------
# Small HTML components
# ----------------------------------------------------------------------------
def aqi_chip(aqi_value, label: str | None = None) -> str:
    cat = A.category(aqi_value)
    txt = label or (f"{aqi_value:.0f} · {cat}" if aqi_value == aqi_value else "No data")
    return f'<span class="vs-aqi" style="background:{A.color_hex(aqi_value)};color:{A.text_color(aqi_value)}">{esc(txt)}</span>'


def level_chip(level: str) -> str:
    from .alerts import LEVEL_COLOR
    return f'<span class="vs-level" style="background:{LEVEL_COLOR.get(level, "#666")}">{esc(level.upper())}</span>'


def prob_bar(p: float, color: str | None = None) -> str:
    color = color or (STATUS["critical"] if p >= 0.7 else STATUS["serious"] if p >= 0.4 else "#8aa4c8")
    return f'<div class="vs-bar"><span style="width:{max(2, min(100, p * 100)):.0f}%;background:{color}"></span></div>'


def kpi(label: str, value, sub: str = "") -> str:
    return f'<div class="vs-kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)


def page_header(title: str, subtitle: str, state=None) -> None:
    chips = provenance_chips(state) if state is not None else ""
    st.markdown(f'<div class="vs-hero"><div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'
                f'<div class="vs-chips">{chips}</div></div>', unsafe_allow_html=True)


KIND_COLOR = {"real": "#0ca30c", "modeled": "#2a78d6", "model": "#2a78d6", "simulated": "#eda100"}
KIND_TEXT = {"real": "Real", "modeled": "Modeled", "model": "Model", "simulated": "Simulated"}


def provenance_chips(state) -> str:
    out = []
    names = {"air_quality": "Air quality", "weather": "Weather", "fires": "Fires", "citizen": "Citizen data", "cpcb_live": "CPCB live"}
    for key, meta in state.sources.items():
        if key == "cpcb_live" and not meta.get("ok"):
            continue
        kind = meta["kind"] if meta.get("ok", True) else "simulated"
        out.append(f'<span class="vs-chip" title="{esc(meta["label"])}"><span class="vs-dot" style="background:{KIND_COLOR[kind]}"></span>'
                   f'{esc(names.get(key, key))}: {KIND_TEXT[kind]}</span>')
    return "".join(out)


def legend_html(items: list[tuple[str, str]]) -> str:
    return '<div class="vs-legend">' + "".join(
        f'<span><i class="vs-sw" style="background:{c}"></i>{esc(t)}</span>' for t, c in items) + "</div>"


def aqi_legend() -> str:
    return legend_html([(f"{n} ({lo}-{hi})", c) for n, lo, hi, c, _ in A.CATEGORIES])


# ----------------------------------------------------------------------------
# Session & secrets
# ----------------------------------------------------------------------------
SECRET_KEYS = ["FIRMS_MAP_KEY", "DATA_GOV_IN_API_KEY", "THINGSPEAK_CHANNEL_ID", "THINGSPEAK_READ_KEY", "VAYUSETU_WEBHOOK_URL"]


def get_secrets() -> dict:
    out = {}
    for k in SECRET_KEYS:
        v = os.environ.get(k)
        try:
            v = st.secrets.get(k, v)
        except Exception:  # noqa: BLE001 - no secrets file
            pass
        if v:
            out[k] = str(v)
    return out


def init_session() -> None:
    ss = st.session_state
    ss.setdefault("mode", "replay")
    ss.setdefault("replay_time", config.REPLAY_DEFAULT_TIME)
    ss.setdefault("city", "Delhi")
    ss.setdefault("user_reports", [])
    ss.setdefault("alert_status", {})
    ss.setdefault("alert_log", [])


def current_key() -> tuple[str, str]:
    ss = st.session_state
    if ss["mode"] == "live":
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
        return "live", str(now)
    return "replay", str(pd.Timestamp(ss["replay_time"]).floor("h"))


# ----------------------------------------------------------------------------
# Cached data access
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False, max_entries=24)
def get_state(mode: str, now_str: str) -> engine.PlatformState:
    if mode == "live":
        return engine.build_live_state(get_secrets())
    return engine.build_replay_state(now_str)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=24)
def get_forecasts(mode: str, now_str: str) -> dict:
    return engine.forecast_all(get_state(mode, now_str))


def _reports_frame(city: str | None = None) -> pd.DataFrame:
    rows = st.session_state.get("user_reports", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df[df["city"] == city] if city else df


@st.cache_data(ttl=1800, show_spinner=False, max_entries=64)
def _hotspots_cached(mode: str, now_str: str, city: str, reports_json: str) -> dict:
    state = get_state(mode, now_str)
    extra = pd.read_json(io.StringIO(reports_json), orient="records") if reports_json != "[]" else None
    if extra is not None and len(extra):
        extra["time"] = pd.to_datetime(extra["time"])
    return engine.city_hotspots(state, city, extra)


def get_hotspots(mode: str, now_str: str, city: str) -> dict:
    rep = _reports_frame(city)
    rj = rep.to_json(orient="records", date_format="iso") if len(rep) else "[]"
    return _hotspots_cached(mode, now_str, city, rj)


def hotspot_cities(state) -> list[str]:
    wc = [c for c in config.ward_cities() if c in state.frames]
    return wc


def get_all_hotspots(mode: str, now_str: str) -> dict:
    state = get_state(mode, now_str)
    return {c: get_hotspots(mode, now_str, c) for c in hotspot_cities(state)}


def get_alerts(mode: str, now_str: str, include_hotspots: bool = True):
    state = get_state(mode, now_str)
    fcs = get_forecasts(mode, now_str)
    hs = get_all_hotspots(mode, now_str) if include_hotspots else {}
    return engine.build_alerts(state, fcs, hs)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads((config.ROOT / path).read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: str, parse_dates: tuple = ()) -> pd.DataFrame:
    return pd.read_csv(config.ROOT / path, parse_dates=list(parse_dates))


# ----------------------------------------------------------------------------
# Plotly styling
# ----------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 360, legend: bool = True, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        margin=margin or dict(l=10, r=10, t=36, b=10),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11, color=INK2),
                    bgcolor="rgba(0,0,0,0)") if legend else None,
        showlegend=legend,
        title_font=dict(size=14, color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=BASE, tickcolor=BASE, ticks="outside", zeroline=False,
                     tickfont=dict(color=MUTED), automargin=True, title_font=dict(size=12, color=INK2))
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=BASE, tickfont=dict(color=MUTED),
                     automargin=True, title_font=dict(size=12, color=INK2))
    return fig


def show_fig(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def aqi_bands(fig: go.Figure, ymax: float, x0, x1, label_side: str = "right") -> None:
    """Very light PM2.5 category bands (NAQI) behind a concentration chart."""
    edges = [0, 30, 60, 90, 120, 250, max(ymax, 260)]
    for (name, *_ , col, _t), lo, hi in zip(A.CATEGORIES, edges[:-1], edges[1:]):
        if lo >= ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=col, opacity=0.08, line_width=0, layer="below")
        fig.add_annotation(x=1.0, xref="paper", y=(lo + min(hi, ymax)) / 2, text=name, showarrow=False,
                           xanchor="left", font=dict(size=10, color=MUTED))


# ----------------------------------------------------------------------------
# Forecast chart
# ----------------------------------------------------------------------------
def forecast_figure(state, city: str, fc: dict, show_actual: bool = True) -> go.Figure:
    hist = state.frames[city]["pm25"].loc[state.now - pd.Timedelta(hours=72): state.now]
    h = fc["hourly"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(h["time"]) + list(h["time"][::-1]), y=list(h["p90"]) + list(h["p10"][::-1]),
                             fill="toself", fillcolor="rgba(235,104,52,0.13)", line=dict(width=0),
                             hoverinfo="skip", name="80% forecast range"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Observed",
                             line=dict(color=INK, width=2), hovertemplate="%{x|%d %b %H:%M}<br>Observed %{y:.0f} µg/m³<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["time"], y=h["p50"], mode="lines", name="VayuCast forecast (median)",
                             line=dict(color=SERIES[1], width=2),
                             hovertemplate="%{x|%d %b %H:%M}<br>Forecast %{y:.0f} µg/m³<extra></extra>"))
    if "physics_pm25" in h:
        fig.add_trace(go.Scatter(x=h["time"], y=h["physics_pm25"], mode="lines", name="Copernicus CAMS (physics)",
                                 line=dict(color=SERIES[0], width=2),
                                 hovertemplate="%{x|%d %b %H:%M}<br>CAMS %{y:.0f} µg/m³<extra></extra>"))
    fut = state.future_obs.get(city)
    if show_actual and fut is not None and not fut.empty:
        fig.add_trace(go.Scatter(x=fut.index, y=fut["pm25"], mode="lines", name="What actually happened",
                                 line=dict(color=SERIES[2], width=2),
                                 hovertemplate="%{x|%d %b %H:%M}<br>Actual %{y:.0f} µg/m³<extra></extra>"))
    ymax = float(np.nanmax([hist.max() if len(hist) else 0, h["p90"].max(),
                            fut["pm25"].max() if fut is not None and len(fut) else 0])) * 1.08
    aqi_bands(fig, ymax, hist.index.min(), h["time"].max())
    fig.add_vline(x=state.now, line_width=1, line_color=BASE)
    fig.add_annotation(x=state.now, y=ymax * 0.97, yref="y", text=" now", showarrow=False, yanchor="top",
                       xanchor="left", font=dict(size=10, color=MUTED))
    fig.update_yaxes(title_text="PM2.5 (µg/m³, hourly)", range=[0, ymax])
    style_fig(fig, height=390, margin=dict(l=10, r=80, t=40, b=10))
    return fig


def corridor_matrix_figure(state, forecasts: dict) -> go.Figure:
    """Heatmap: corridor cities x next 72 h (6-hourly), coloured by AQI category of forecast PM2.5."""
    rows, labels = [], []
    for code in config.MAJOR_CORRIDORS:
        for city in config.CORRIDORS[code]["path"]:
            if city in forecasts and f"{city}" not in [l.split(" · ")[0] for l in labels]:
                h = forecasts[city]["hourly"].set_index("time")["p50"]
                six = h.resample("6h", origin=h.index[0]).mean()
                rows.append(six)
                labels.append(f"{city} · {code}")
    if not rows:
        return go.Figure()
    cols = rows[0].index
    Z = np.array([[A.category_index(A.subindex("pm25", v)) for v in r.reindex(cols).values] for r in rows], float)
    V = np.array([r.reindex(cols).values for r in rows])
    colorscale = []
    n = len(A.CATEGORIES)
    for i, (_, _, _, col, _) in enumerate(A.CATEGORIES):
        colorscale += [[i / n, col], [(i + 1) / n, col]]
    text = [[f"{A.category(A.subindex('pm25', v))}<br>{v:.0f} µg/m³" if v == v else "" for v in r] for r in V]
    fig = go.Figure(go.Heatmap(z=Z, x=[t.strftime("%d %b %H:%M") for t in cols], y=labels, zmin=0, zmax=n,
                               colorscale=colorscale, showscale=False, xgap=2, ygap=2, text=text,
                               hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11, color=INK2))
    style_fig(fig, height=max(260, 24 * len(labels) + 60), legend=False, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickangle=0, dtick=2, ticks="")
    return fig


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------
def records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Only the columns a map layer needs, with NaN -> None (strict JSON for deck.gl)."""
    sub = df[[c for c in cols if c in df.columns]]
    out = []
    for row in sub.itertuples(index=False):
        rec = {}
        for k, v in zip(sub.columns, row):
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                v = None
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            rec[k] = v
        out.append(rec)
    return out


EXCESS_BINS = [(10, [225, 225, 222, 50], "< 10"), (25, [253, 208, 162, 130], "10-25"), (50, [253, 174, 107, 160], "25-50"),
               (100, [241, 105, 19, 180], "50-100"), (1e9, [166, 54, 3, 200], "> 100")]


def excess_rgb(v) -> list[int]:
    if v is None or not np.isfinite(v):
        return [220, 220, 220, 30]
    for hi, col, _ in EXCESS_BINS:
        if v < hi:
            return col
    return EXCESS_BINS[-1][1]


def excess_legend() -> str:
    return legend_html([(f"{lab} µg/m³ above local background", "rgba(%d,%d,%d,%.2f)" % (c[0], c[1], c[2], max(c[3], 90) / 255))
                        for _, c, lab in EXCESS_BINS])


TOOLTIP_STYLE = {"backgroundColor": "white", "color": INK, "fontSize": "12px", "border": f"1px solid {GRID}",
                 "borderRadius": "8px", "padding": "8px", "fontFamily": FONT}


TOOLTIP_HTML = ("<div style='font-weight:650;margin-bottom:2px'>{t}</div><div>{l1}</div><div>{l2}</div>"
                "<div>{l3}</div><div>{l4}</div>")
TIP_COLS = ["t", "l1", "l2", "l3", "l4"]


def tipf(title: str, *lines: str) -> dict:
    """Plain-text tooltip fields (the map renderer escapes values, so no HTML here)."""
    lines = [str(x) for x in lines if x][:4]
    out = {"t": str(title)}
    for i in range(4):
        out[f"l{i + 1}"] = lines[i] if i < len(lines) else ""
    return out


class CompactDeck(pdk.Deck):
    """pydeck serialises with indent=2 (~5x larger); send compact JSON to the browser instead."""

    def to_json(self):
        return json.dumps(json.loads(super().to_json()), separators=(",", ":"))


def dots(data, px: float, fill, line=None, line_px: float = 1.0, pickable: bool = True, pos=("lon", "lat")) -> pdk.Layer:
    """Scatter layer sized in screen pixels, clamped so every renderer draws the same size."""
    kw = dict(get_position=list(pos), get_fill_color=fill, get_radius=px, radius_units="pixels",
              radius_min_pixels=px, radius_max_pixels=px, pickable=pickable)
    if line is not None:
        kw.update(stroked=True, get_line_color=line, line_width_min_pixels=line_px, line_width_max_pixels=line_px)
    return pdk.Layer("ScatterplotLayer", data, **kw)


def m_per_px(lat: float, zoom: float) -> float:
    return 156543.03392 * np.cos(np.radians(lat)) / (2 ** zoom)


def _deck(layers, view: pdk.ViewState, height: int = 520) -> pdk.Deck:
    return CompactDeck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                       tooltip={"html": TOOLTIP_HTML, "style": TOOLTIP_STYLE})


def _states_layer(alpha: int = 18):
    gj = json.loads(json.dumps(geo.india_states()))
    for f in gj["features"]:
        f["properties"].update(tipf(f["properties"]["name"]))
    return pdk.Layer("GeoJsonLayer", gj, stroked=True, filled=True, get_fill_color=[120, 130, 140, alpha],
                     get_line_color=[120, 120, 120, 150], line_width_min_pixels=0.6, pickable=False)


def wind_segments(state, cities: list[str], km_per_ms: float = 25.0, min_km: float = 12.0) -> list[dict]:
    segs = []
    for c in cities:
        ws, wd, _ = engine.wind_now(state, c)
        if np.isnan(ws):
            continue
        meta = config.CITIES[c]
        to = np.radians((wd + 180) % 360)
        L = max(min_km, ws * km_per_ms)
        lat2 = meta.lat + L * np.cos(to) / 111.32
        lon2 = meta.lon + L * np.sin(to) / (111.32 * np.cos(np.radians(meta.lat)))
        segs.append({"source": [meta.lon, meta.lat], "target": [lon2, lat2],
                     **tipf(c, f"Wind from {wd:.0f}° at {ws:.1f} m/s")})
    return segs


def india_map(state, forecasts: dict, show_fires: bool = True, show_corridors: bool = True,
              show_wind: bool = True, height: int = 560) -> pdk.Deck:
    snap = state.snapshot.copy()
    layers = [_states_layer()]
    if show_corridors:
        paths = []
        for code in config.MAJOR_CORRIDORS:
            meta = config.CORRIDORS[code]
            pts = [[config.CITIES[c].lon, config.CITIES[c].lat] for c in meta["path"]]
            paths.append({"path": pts, "color": meta["color"] + [150], **tipf(meta["name"], " → ".join(meta["path"]))})
        layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_color="color", width_min_pixels=3,
                                width_scale=1, pickable=True, cap_rounded=True, joint_rounded=True))
    if show_fires and state.fires is not None and len(state.fires):
        f = state.fires[["longitude", "latitude", "frp"]].copy()
        f["frp"] = f["frp"].fillna(5).clip(1, 200).round(1)
        f[["longitude", "latitude"]] = f[["longitude", "latitude"]].round(4)
        layers.append(pdk.Layer("HeatmapLayer", records(f, ["longitude", "latitude", "frp"]),
                                get_position=["longitude", "latitude"], get_weight="frp",
                                radius_pixels=16, intensity=0.6, threshold=0.12, opacity=0.35,
                                color_range=[[254, 237, 222], [253, 190, 133], [253, 141, 60], [230, 85, 13], [166, 54, 3]]))
    if show_wind:
        segs = wind_segments(state, list(snap["city"]), km_per_ms=55.0, min_km=70.0)
        layers.append(pdk.Layer("LineLayer", segs, get_source_position="source", get_target_position="target",
                                get_color=[40, 40, 40, 190], get_width=2, width_min_pixels=2, pickable=True))
    rows = []
    for _, r in snap.iterrows():
        fc = forecasts.get(r["city"])
        worst = max(fc["windows"], key=lambda w: w["aqi"]) if fc else None
        pvp = max(w["prob_very_poor"] for w in fc["windows"]) if fc else float("nan")
        tip = tipf(f"{r['city']} · {r['state']}", f"AQI now {r['aqi']:.0f} ({r['category']})",
                   f"24-h PM2.5 {r['pm25_24h']:.0f} µg/m³",
                   f"Worst next 72 h: AQI {worst['aqi']:.0f} ({worst['category']})" if worst else "",
                   f"P(Very Poor or worse): {pvp:.0%}" if worst else "")
        c = config.CITIES[r["city"]]
        rows.append({"city": r["city"], "lon": r["lon"], "lat": r["lat"], "color": A.color_rgb(r["aqi"], 235),
                     "radius": 12000 + 4500 * np.sqrt(c.population_m), **tip,
                     "label": r["city"] if (c.population_m >= 2.4 or r["aqi"] > 300) and r["city"] != "Gurugram" else ""})
    df = pd.DataFrame(rows)
    pops = df["city"].map(lambda x: config.CITIES[x].population_m)
    for lo, hi, px in ((0, 2.0, 7), (2.0, 9.0, 10), (9.0, 1e9, 13)):
        sub = df[(pops >= lo) & (pops < hi)]
        if len(sub):
            layers.append(dots(records(sub, ["lon", "lat", "color", *TIP_COLS]), px, "color", [255, 255, 255, 255], 2))
    layers.append(pdk.Layer("TextLayer", records(df[df["label"] != ""], ["lon", "lat", "label"]), get_position=["lon", "lat"],
                            get_text="label", get_size=12,
                            get_color=[30, 30, 30, 235], get_pixel_offset=[0, -17], get_text_anchor="'middle'",
                            font_family="'Arial, Helvetica, sans-serif'", font_weight=600,
                            outline_width=3, outline_color=[252, 252, 251, 255], font_settings={"sdf": True}))
    view = pdk.ViewState(latitude=21.3, longitude=81.0, zoom=3.55, pitch=0)
    return _deck(layers, view, height)


SENSOR_STATUS_COLOR = {"normal": None, "hotspot": [211, 47, 47, 255], "offline": [150, 150, 150, 180],
                       "stuck": [20, 20, 20, 255], "suspect-drift": [20, 20, 20, 255]}


def city_map(state, res: dict, show_wards: bool = True, show_sensors: bool = True, show_hotspots: bool = True,
             show_reports: bool = True, show_pois: bool = True, satellite: str | None = None,
             ward_mode: str = "excess") -> pdk.Deck:
    city = res["city"]
    c = config.CITIES[city]
    layers = []
    if satellite:
        lat_span, lon_span = 0.35, 0.45
        bbox = (c.lon - lon_span, c.lat - lat_span, c.lon + lon_span, c.lat + lat_span)
        day = (state.now - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        url = live.gibs_wms_url(live.GIBS_LAYERS[satellite], day, bbox, fmt="image/png" if "AOD" in satellite else "image/jpeg")
        layers.append(pdk.Layer("BitmapLayer", image=url, bounds=[bbox[0], bbox[1], bbox[2], bbox[3]], opacity=0.75))
    if show_wards and res.get("wards"):
        feats = []
        wv, wx = res["ward_values"], res.get("ward_excess", {})
        for f in res["wards"]["features"]:
            p = f["properties"]
            v, e = wv.get(p["ward_id"]), wx.get(p["ward_id"])
            if ward_mode == "excess":
                fill = excess_rgb(e)
            else:
                fill = A.pm25_to_rgb(v, 80) if v is not None else [200, 200, 200, 40]
            tip = tipf(p["ward_name"], f"Ward {p['ward_id']}",
                       f"PM2.5 (3-h, sensors) ≈ {v:.0f} µg/m³ · {A.category(A.subindex('pm25', v))}" if v is not None else "no data",
                       f"Local excess ≈ {e:+.0f} µg/m³" if e is not None else "")
            feats.append({"type": "Feature", "geometry": f["geometry"], "properties": {"fill": fill, **tip}})
        layers.append(pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": feats}, stroked=True, filled=True,
                                get_fill_color="properties.fill", get_line_color=[255, 255, 255, 230],
                                line_width_min_pixels=0.8, pickable=True, auto_highlight=True))
    if show_sensors:
        s = res["sensors"].copy()
        s["color"] = [A.pm25_to_rgb(v, 235) if st_ == "normal" else SENSOR_STATUS_COLOR[st_]
                      for st_, v in zip(s["status"], s["pm25"].fillna(0))]
        tips = [tipf(f"Sensor {i} · {h}", str(w), f"Status: {stt}",
                     f"PM2.5 (3-h) {v:.0f} µg/m³" if v == v else "no recent data",
                     f"Excess over local background: {e:+.0f} µg/m³" if e == e else "")
                for i, h, w, stt, v, e in zip(s["sensor_id"], s["host"], s["ward_name"], s["status"], s["pm25"], s["excess"])]
        for k in TIP_COLS:
            s[k] = [tp[k] for tp in tips]
        normal = s[s["status"] == "normal"]
        other = s[s["status"] != "normal"]
        layers.append(dots(records(normal, ["lon", "lat", "color", *TIP_COLS]), 3.5, "color", [255, 255, 255, 255], 1))
        if len(other):
            layers.append(dots(records(other, ["lon", "lat", "color", *TIP_COLS]), 5.5, "color", [255, 255, 255, 255], 1.5))
    if show_pois:
        pois = pd.DataFrame([{"lon": p.lon, "lat": p.lat, "label": p.name,
                              **tipf(p.name, f"{p.kind.title()} (reference location)")}
                             for p in config.POIS.get(city, [])])
        if len(pois):
            recs = records(pois, ["lon", "lat", "label", *TIP_COLS])
            layers.append(dots(recs, 4, [70, 70, 70, 230], [255, 255, 255, 255], 1))
            layers.append(pdk.Layer("TextLayer", recs, get_position=["lon", "lat"], get_text="label", get_size=11,
                                    get_color=[50, 50, 50, 235], get_pixel_offset=[0, 12], get_text_anchor="'middle'",
                                    font_family="'Arial, Helvetica, sans-serif'",
                                    outline_width=3, outline_color=[252, 252, 251, 255], font_settings={"sdf": True}))
    if show_reports and len(res["reports"]):
        r = res["reports"]
        r = r[(r["time"] > state.now - pd.Timedelta(hours=12)) & (r["time"] <= state.now + pd.Timedelta(minutes=59))].copy()
        if len(r):
            r["color"] = [hex_rgb(SOURCE_COLORS.get(k, "#9e9e9e"), 255) for k in r["kind"]]
            tips = [tipf("📷 Citizen report", str(cat), f"{pd.Timestamp(t):%d %b %H:%M}", str(n))
                    for cat, t, n in zip(r["category"], r["time"], r["note"])]
            for k in TIP_COLS:
                r[k] = [tp[k] for tp in tips]
            layers.append(dots(records(r, ["lon", "lat", "color", *TIP_COLS]), 6, [255, 255, 255, 255], "color", 3))
    if state.fires is not None and len(state.fires):
        f = state.fires
        near = f[(abs(f["latitude"] - c.lat) < 0.5) & (abs(f["longitude"] - c.lon) < 0.6)].copy()
        if len(near):
            tips = [tipf("🔥 Satellite fire detection", str(s_), f"Fire radiative power {p:.1f} MW")
                    for s_, p in zip(near["satellite"], near["frp"].fillna(0))]
            for k in TIP_COLS:
                near[k] = [tp[k] for tp in tips]
            layers.append(dots(records(near, ["longitude", "latitude", *TIP_COLS]), 4, [230, 85, 13, 230],
                               pos=("longitude", "latitude")))
    if show_hotspots and res["hotspots"]:
        h = pd.DataFrame([{"lon": x["lon"], "lat": x["lat"], "radius": x["radius_km"] * 1000,
                           "color": hex_rgb(SOURCE_COLORS[x["source"]], 45), "line": hex_rgb(SOURCE_COLORS[x["source"]], 255),
                           **tipf(f"{config.SOURCE_TYPES[x['source']]['icon']} Hotspot · {x['source_label']}",
                                  f"+{x['peak_excess']:.0f} µg/m³ over local background",
                                  f"Confidence {x['confidence']:.0%} · {x['severity']}", ", ".join(x["wards"][:3]))}
                          for x in res["hotspots"]])
        mpp = m_per_px(c.lat, zoom_for(city))
        for rec in records(h, ["lon", "lat", "radius", "color", "line", *TIP_COLS]):
            px = float(np.clip(rec["radius"] / mpp, 13, 70))
            layers.append(dots([rec], px, rec["color"], rec["line"], 3))
    segs = wind_segments(state, [city], km_per_ms=1.6)
    if segs:
        layers.append(pdk.Layer("LineLayer", segs, get_source_position="source", get_target_position="target",
                                get_color=[30, 30, 30, 200], get_width=3, width_min_pixels=3, pickable=True))
    return _deck(layers, pdk.ViewState(latitude=c.lat, longitude=c.lon, zoom=zoom_for(city), pitch=0))


def zoom_for(city: str) -> float:
    if city == "Delhi":
        return 9.75
    return 10.0 if city in ("Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Mumbai") else 10.8


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
KEY_MOMENTS = {
    "30 Oct 22:00 · smog building": "2019-10-30 22:00",
    "1 Nov 18:00 · health emergency": "2019-11-01 18:00",
    "3 Nov 14:00 · peak": "2019-11-03 14:00",
    "6 Nov 12:00 · winds clear": "2019-11-06 12:00",
}


def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown('<div class="vs-brand">🌬️ VayuSetu</div><div class="vs-tag">Federated clean-air intelligence '
                    'for Indian cities</div>', unsafe_allow_html=True)
        st.divider()
        mode = st.segmented_control("Data mode", ["Replay 2019", "Live"], default="Replay 2019" if ss["mode"] == "replay" else "Live",
                                    key="mode_picker", required=True, help="Replay uses real CPCB data from the Oct-Nov 2019 Delhi-NCR smog "
                                    "episode. Live pulls Copernicus CAMS, weather and NASA satellite fires right now.")
        ss["mode"] = "live" if mode == "Live" else "replay"
        if ss["mode"] == "replay":
            opts = [str(t) for t in pd.date_range(config.REPLAY_START, config.REPLAY_END, freq="h")]
            if "replay_slider" not in ss:
                ss["replay_slider"] = str(pd.Timestamp(ss["replay_time"]))
            moment = st.pills("Key moments", list(KEY_MOMENTS), selection_mode="single", key="moment_pills")
            if moment and ss.get("_last_moment") != moment:
                ss["_last_moment"] = moment
                ss["replay_slider"] = str(pd.Timestamp(KEY_MOMENTS[moment]))
            st.select_slider("Replay time (IST)", options=opts, key="replay_slider",
                             format_func=lambda s: pd.Timestamp(s).strftime("%d %b %Y · %H:%M"))
            ss["replay_time"] = ss["replay_slider"]
        else:
            st.caption("Live data refreshes every 30 minutes. Optional keys (FIRMS, data.gov.in, ThingSpeak, "
                       "webhook) are read from Streamlit secrets.")
        st.divider()
        st.caption("Built for the Clean Air & Climate Resilience challenge. Data: CPCB, Copernicus CAMS, "
                   "Open-Meteo, NASA FIRMS/GIBS, DataMeet boundaries. Citizen-sensor data is simulated for "
                   "the pilot design.")
