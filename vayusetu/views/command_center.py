"""National Command Center: current air quality, 72-h spike risk, fires, alerts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vayusetu import aqi as A
from vayusetu import config, engine, ui

mode, now_str = ui.current_key()
with st.spinner("Building the national picture (data fusion, forecasts, hotspot scans)..."):
    state = ui.get_state(mode, now_str)
    fcs = ui.get_forecasts(mode, now_str)
    hs_all = ui.get_all_hotspots(state.mode, now_str)
    alerts = engine.build_alerts(state, fcs, hs_all)

label = "Replay · Delhi-NCR smog episode 2019" if state.mode == "replay" else "Live"
ui.page_header("National Clean-Air Command Center",
               f"{label} · {state.now:%a %d %b %Y, %H:%M} IST · {len(state.snapshot)} cities across India's economic corridors",
               state)
for n in state.notices[:2]:
    st.caption(f"ℹ️ {n}")

snap = state.snapshot
bad_now = snap[snap["aqi"] > 300]
risk = {c: max(w["prob_very_poor"] for w in fc["windows"]) for c, fc in fcs.items()}
risky = [c for c, p in risk.items() if p >= config.THRESHOLDS.spike_prob_alert]
n_hot = sum(1 for r in hs_all.values() for h in r["hotspots"] if h["confidence"] >= 0.6)
lv = pd.Series([a.level for a in alerts]).value_counts() if alerts else pd.Series(dtype=int)
fires_by_state = state.fires["state"].value_counts() if len(state.fires) else pd.Series(dtype=int)
worst = snap.sort_values("aqi", ascending=False).iloc[0]

ui.kpi_row([
    ("Cities Very Poor or Severe now", f"{len(bad_now)} / {len(snap)}", ", ".join(bad_now.sort_values("aqi", ascending=False)["city"].head(3)) or "none"),
    ("Spike risk in next 72 h", f"{len(risky)} cities", "P(Very Poor or worse) ≥ 55%"),
    ("Hidden hotspots found", f"{n_hot}", f"ward-level scans in {len(hs_all)} cities"),
    ("Active alerts", f"{len(alerts)}", f"{lv.get('Emergency', 0)} emergency · {lv.get('Warning', 0)} warning · {lv.get('Advisory', 0)} advisory"),
    ("Fire detections (24 h)" if state.mode == "replay" else "Satellite fire detections (24 h)", f"{len(state.fires):,}",
     (" · ".join(f"{k} {v:,}" for k, v in fires_by_state.head(2).items()) or "none")
     + (" · simulated for replay" if state.mode == "replay" else " · NASA VIIRS")),
    ("Worst air now", f"{worst['city']}", f"AQI {worst['aqi']:.0f} · {worst['category']}"),
])

st.write("")
left, right = st.columns([2.25, 1], gap="medium")
with left:
    t1, t2, t3 = st.columns(3)
    show_fires = t1.toggle("Satellite fires", value=True)
    show_corr = t2.toggle("Economic corridors", value=True)
    show_wind = t3.toggle("Wind (downwind lines)", value=True)
    st.pydeck_chart(ui.india_map(state, fcs, show_fires, show_corr, show_wind), height=560)
    st.markdown(ui.aqi_legend(), unsafe_allow_html=True)
    st.caption("Bubbles: city AQI (24-h NAQI, size = population). Orange heat: satellite fire detections, last 24 h. "
               "Black lines point downwind. Coloured paths: industrial corridors. Hover for details.")

with right:
    st.markdown('<div class="vs-section">Highest spike risk · next 72 h</div>', unsafe_allow_html=True)
    top = sorted(fcs.items(), key=lambda kv: -max(w["prob_very_poor"] for w in kv[1]["windows"]))[:6]
    for city, fc in top:
        w = max(fc["windows"], key=lambda x: x["aqi"])
        pvp = max(x["prob_very_poor"] for x in fc["windows"])
        ps = max(x["prob_severe"] for x in fc["windows"])
        now_aqi = float(snap.set_index("city").loc[city, "aqi"])
        chip_now = ui.aqi_chip(now_aqi, "Now %.0f" % now_aqi)
        chip_fc = ui.aqi_chip(w["aqi"], "%s: %.0f %s" % (w["window"], w["aqi"], w["category"]))
        corridors = ui.esc(" · ".join(config.CITIES[city].corridors))
        st.markdown(
            f'<div class="vs-card"><div class="vs-row"><b style="color:#0b0b0b;font-size:.95rem">{ui.esc(city)}</b>'
            f'<span>{corridors}</span></div>'
            f'<div style="margin:.35rem 0">{chip_now} → {chip_fc}</div>'
            f'<div class="vs-row"><span>P(Very Poor or worse)</span><b>{pvp:.0%}</b></div>{ui.prob_bar(pvp)}'
            f'<div class="vs-row"><span>P(Severe)</span><b>{ps:.0%}</b></div>{ui.prob_bar(ps)}</div>',
            unsafe_allow_html=True)

st.markdown('<div class="vs-section">Corridor outlook · next 72 hours (6-hourly forecast AQI category)</div>',
            unsafe_allow_html=True)
ui.show_fig(ui.corridor_matrix_figure(state, fcs), key="corridor_matrix")
st.markdown(ui.aqi_legend(), unsafe_allow_html=True)

c1, c2 = st.columns([1.3, 1], gap="medium")
with c1:
    st.markdown('<div class="vs-section">Latest alerts</div>', unsafe_allow_html=True)
    for a in alerts[:6]:
        st.markdown(f'<div class="vs-card">{ui.level_chip(a.level)} <b style="color:#0b0b0b">{ui.esc(a.title)}</b>'
                    f'<div class="meta" style="margin-top:.3rem">{ui.esc(a.city)} · {ui.esc(a.summary[:190])}'
                    f'{"..." if len(a.summary) > 190 else ""}</div></div>', unsafe_allow_html=True)
    st.page_link("views/alerts.py", label="Open Alerts & Response", icon="🚨")
with c2:
    st.markdown('<div class="vs-section">Regional smoke transport</div>', unsafe_allow_html=True)
    if len(fires_by_state):
        fb = fires_by_state.head(6)[::-1]
        fig = go.Figure(go.Bar(x=fb.values, y=fb.index, orientation="h", marker=dict(color=ui.SERIES[1], cornerradius=4),
                               width=0.5, text=[f"{v:,}" for v in fb.values], textposition="outside", cliponaxis=False,
                               textfont=dict(color=ui.INK2), hovertemplate="%{y}: %{x:,} detections<extra></extra>"))
        ui.style_fig(fig, height=max(170, 60 + 42 * len(fb)), legend=False, margin=dict(l=10, r=50, t=10, b=10))
        fig.update_xaxes(title_text="Fire detections, last 24 h", range=[0, float(fb.max()) * 1.18])
        ui.show_fig(fig, key="fires_by_state")
    rows = []
    for city in ["Delhi", "Gurugram", "Chandigarh", "Amritsar", "Lucknow", "Jaipur", "Patna"]:
        if city in state.frames:
            up = engine.regional_transport(state, city)
            rows.append({"City": city, "Wind from": f"{up['wind_dir']:.0f}°" if up["wind_dir"] == up["wind_dir"] else "-",
                         "Wind (m/s)": round(up["wind_speed"], 1), "Upwind fires (24 h)": up["count"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    src = state.sources["fires"]
    st.caption(f"Fire source: {src['label']}. Upwind = within {config.THRESHOLDS.fire_upwind_km:.0f} km and "
               f"±{config.THRESHOLDS.fire_upwind_half_angle:.0f}° of the wind direction.")

with st.expander("View data: city snapshot"):
    show = snap[["city", "state", "corridors", "pm25_now", "pm25_24h", "aqi", "category"]].copy()
    show["P(Very Poor+) 72h"] = show["city"].map(risk)
    st.dataframe(show.sort_values("aqi", ascending=False).round(1), hide_index=True)
