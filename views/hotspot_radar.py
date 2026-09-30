"""Hotspot Radar: ward-level detection of hidden pollution sources in one city."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vayusetu import aqi as A
from vayusetu import config, engine, ui
from vayusetu.data import live

mode, now_str = ui.current_key()
ss = st.session_state
state = ui.get_state(mode, now_str)

ward_first = ui.hotspot_cities(state)
others = sorted(c for c in state.frames if c not in ward_first)
cities = ward_first + others
if ss.get("city") not in cities:
    ss["city"] = cities[0]
top_l, top_r = st.columns([1, 3])
city = top_l.selectbox("City", cities, index=cities.index(ss["city"]), key="radar_city",
                       format_func=lambda c: f"{c} (ward map)" if c in ward_first else c)
ss["city"] = city

with st.spinner(f"Scanning {city}'s citizen-sensor network..."):
    res = ui.get_hotspots(state.mode, now_str, city)
hs = res["hotspots"]
sensors = res["sensors"]
ws, wd, rh = engine.wind_now(state, city)

ui.page_header(f"Hotspot Radar · {city}",
               f"Hyper-local scan at {state.now:%d %b %Y, %H:%M} IST · 3-hour window · "
               f"{len(sensors)} citizen sensors · wind from {wd:.0f}° at {ws:.1f} m/s", state)

counts = sensors["status"].value_counts()
confident = [h for h in hs if h["confidence"] >= 0.6]
bg = float(np.nanmedian(sensors["bg_pm25"])) if sensors["bg_pm25"].notna().any() else float("nan")
top_src = pd.Series([h["source_label"] for h in confident]).value_counts()
ui.kpi_row([
    ("Sensors reporting", f"{int(counts.get('normal', 0) + counts.get('hotspot', 0))} / {len(sensors)}",
     f"{int(counts.get('offline', 0))} offline"),
    ("Hotspots (alert-grade)", f"{len(confident)}", f"{len(hs)} candidates incl. low confidence"),
    ("Most common source", top_src.index[0] if len(top_src) else "-", f"{int(top_src.iloc[0])} hotspot(s)" if len(top_src) else ""),
    ("Neighbourhood background", f"{bg:.0f} µg/m³" if bg == bg else "-", A.category(A.subindex("pm25", bg)) if bg == bg else ""),
    ("Faulty sensors quarantined", f"{int(counts.get('stuck', 0) + counts.get('suspect-drift', 0))}", "stuck or drifting units"),
])
st.write("")

c = st.columns([1, 1, 1, 1, 1, 1.6])
show_w = c[0].toggle("Wards", value=res.get("wards") is not None, disabled=res.get("wards") is None)
show_s = c[1].toggle("Sensors", value=True)
show_h = c[2].toggle("Hotspots", value=True)
show_r = c[3].toggle("Photo reports", value=True)
show_p = c[4].toggle("Known sites", value=True)
sat = c[5].selectbox("Satellite overlay (NASA GIBS)", ["None", *list(live.GIBS_LAYERS)],
                     help="Satellite imagery for yesterday, loaded directly by your browser from NASA GIBS.")

left, right = st.columns([2.05, 1], gap="medium")
with left:
    ward_mode = st.segmented_control("Colour wards by", ["Local excess", "AQI category"], default="Local excess",
                                     required=True, key="ward_mode",
                                     help="Local excess shows where air is dirtier than the surrounding neighbourhood - "
                                          "hidden hotspots stand out even on a city-wide smog day.")
    deck = ui.city_map(state, res, show_w, show_s, show_h, show_r, show_p, None if sat == "None" else sat,
                       ward_mode="excess" if ward_mode == "Local excess" else "aqi")
    st.pydeck_chart(deck, height=600)
    st.markdown(ui.excess_legend() if ward_mode == "Local excess" else ui.aqi_legend(), unsafe_allow_html=True)
    st.markdown(
        ui.legend_html([(config.SOURCE_TYPES[k]["label"] + " hotspot", v) for k, v in ui.SOURCE_COLORS.items()]
                       + [("Faulty sensor", "#141414"), ("Offline", "#969696")]), unsafe_allow_html=True)
    st.caption("Dots: citizen sensors (colour = PM2.5 category). Rings: detected hotspots, coloured by likely source. "
               "White dots with a coloured ring: citizen photo reports. Grey dots: known industrial areas, landfills "
               "and junctions. Black line: wind direction (downwind).")

with right:
    st.markdown(f'<div class="vs-section">Detected hotspots ({len(hs)})</div>', unsafe_allow_html=True)
    if not hs:
        st.info("No local hotspots stand out from the neighbourhood background right now.")
    for h in hs[:7]:
        src = h["source"]
        probs = sorted(h["source_probs"].items(), key=lambda kv: -kv[1])
        bars = "".join(
            f'<div class="vs-row"><span>{ui.esc(config.SOURCE_TYPES[k]["label"])}</span><b>{p:.0%}</b></div>'
            f'{ui.prob_bar(p, ui.SOURCE_COLORS[k])}' for k, p in probs[:3])
        ev = "".join(f"<li>{ui.esc(e)}</li>" for e in h["evidence"][:4])
        conf_txt = "alert-grade" if h["confidence"] >= 0.6 else "watch"
        st.markdown(
            f'<div class="vs-card"><h4>{config.SOURCE_TYPES[src]["icon"]} {ui.esc(h["source_label"])}</h4>'
            f'<div class="meta">{ui.esc(", ".join(h["wards"][:3]))}{" ..." if len(h["wards"]) > 3 else ""}</div>'
            f'<div style="margin:.4rem 0">{ui.aqi_chip(A.subindex("pm25", h["peak_pm25"]), "Peak %.0f µg/m³" % h["peak_pm25"])} '
            f'<span class="vs-chip">+{h["peak_excess"]:.0f} over local</span> '
            f'<span class="vs-chip">{ui.esc(h["severity"])}</span> '
            f'<span class="vs-chip">{h["confidence"]:.0%} {conf_txt}</span></div>{bars}'
            f'<ul>{ev}</ul><div class="vs-note">{h["n_sensors"]} sensor(s) · Gi* z = {h["gi_z"]:.1f} · '
            f'elevated ~{h["persistence_h"]} h</div></div>', unsafe_allow_html=True)

if hs:
    st.markdown('<div class="vs-section">Inspect a hotspot</div>', unsafe_allow_html=True)
    opts = {f"{config.SOURCE_TYPES[h['source']]['icon']} {h['source_label']} · {', '.join(h['wards'][:2])} "
            f"(+{h['peak_excess']:.0f})": h for h in hs}
    pick = st.selectbox("Hotspot", list(opts), label_visibility="collapsed")
    h = opts[pick]
    a, b = st.columns([1.6, 1], gap="medium")
    with a:
        rd = res["readings"]
        peak = max(h["sensor_ids"], key=lambda s: float(sensors.set_index("sensor_id").loc[s, "excess"]))
        win = rd[rd["time"] > state.now - pd.Timedelta(hours=36)]
        piv = win.pivot_table(index="time", columns="sensor_id", values="pm25_cal", aggfunc="mean")
        ok = [s for s in piv.columns if s in set(sensors.loc[sensors["flag"] == "ok", "sensor_id"])]
        bgline = piv[ok].quantile(0.35, axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bgline.index, y=bgline.values, name="Neighbourhood background (35th pct.)",
                                 line=dict(color=ui.MUTED, width=2), hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} µg/m³<extra></extra>"))
        fig.add_trace(go.Scatter(x=piv.index, y=piv[peak], name=f"Sensor {peak} (hotspot core)",
                                 line=dict(color=ui.SOURCE_COLORS[h["source"]], width=2),
                                 hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} µg/m³<extra></extra>"))
        fig.add_vline(x=state.now, line_width=1, line_color=ui.BASE)
        fig.update_yaxes(title_text="PM2.5 (µg/m³, calibrated)")
        ui.style_fig(fig, height=320)
        fig.update_layout(title=dict(text="Last 36 h at the hotspot core vs. its neighbourhood", x=0, font=dict(size=13)))
        ui.show_fig(fig, key="hs_ts")
    with b:
        pr = pd.Series(h["source_probs"]).sort_values()
        fig = go.Figure(go.Bar(x=pr.values, y=[config.SOURCE_TYPES[k]["label"] for k in pr.index], orientation="h",
                               marker=dict(color=[ui.SOURCE_COLORS[k] for k in pr.index], cornerradius=4), width=0.5,
                               text=[f"{v:.0%}" for v in pr.values], textposition="outside", textfont=dict(color=ui.INK2),
                               hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        ui.style_fig(fig, height=250, legend=False, margin=dict(l=10, r=40, t=36, b=10))
        fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
        fig.update_layout(title=dict(text="Source attribution", x=0, font=dict(size=13)))
        ui.show_fig(fig, key="hs_src")
        st.caption(f"Fingerprint: PM2.5/PM10 excess ratio {h['pm_ratio']:.2f}"
                   + (f" · CO/PM {h['co_per_pm'] * 1000:.0f}" if h["co_per_pm"] == h["co_per_pm"] else "")
                   + (f" · NO2/PM {h['no2_per_pm']:.2f}" if h["no2_per_pm"] == h["no2_per_pm"] else ""))
    with st.expander("View data: sensors in this hotspot"):
        st.dataframe(sensors[sensors["sensor_id"].isin(h["sensor_ids"])][
            ["sensor_id", "ward_name", "host", "model", "pm25", "bg_pm25", "excess", "gi_z"]].round(1), hide_index=True)

with st.expander("How detection works, and chronic hotspots"):
    st.markdown(
        "1. **Quality control** removes offline, stuck or drifting sensors.  \n"
        "2. Each sensor's **usual offset** (7-day baseline) is learned, so a sensor that always reads high "
        "near a busy road does not raise false alarms.  \n"
        "3. **Local background** = 35th percentile of sensors within 6 km; a city-wide smog day is not a hotspot.  \n"
        "4. **Getis-Ord Gi\\*** tests whether high excess is spatially clustered; **DBSCAN** groups hot sensors.  \n"
        "5. **Evidence fusion** adds citizen photos, satellite fires and known industrial areas / landfills.  \n"
        "6. **Source attribution** scores chemical fingerprints (PM2.5/PM10, CO, NO2) and time-of-day patterns.")
    chronic = sensors[sensors["chronic"]]
    if len(chronic):
        cw = chronic.groupby("ward_name")["baseline_ratio"].max().sort_values(ascending=False).head(12)
        st.markdown("**Chronic hotspots** - wards whose sensors run persistently above their surroundings "
                    "(candidates for long-term source apportionment):")
        st.dataframe(pd.DataFrame({"Ward": cw.index, "Typical level vs. city": [f"+{(v - 1) * 100:.0f}%" for v in cw.values]}),
                     hide_index=True)
