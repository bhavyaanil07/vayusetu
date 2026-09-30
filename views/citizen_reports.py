"""Citizen Reports: photo evidence with on-device style analysis, the sensor network, and real nodes."""
import io
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from vayusetu import config, ui
from vayusetu.data import citizen, live
from vayusetu.data.citizen import km_offsets
from vayusetu.models import vision

mode, now_str = ui.current_key()
ss = st.session_state
state = ui.get_state(mode, now_str)

ui.page_header("Citizen Reports",
               "Photos and low-cost sensors from residents, schools and shops fill the gaps between official monitors",
               state)

tab1, tab2, tab3 = st.tabs(["📸 Report pollution", "📡 Citizen sensor network", "🔌 Connect a real sensor"])

# ----------------------------------------------------------------------------
with tab1:
    cities = ui.hotspot_cities(state) + sorted(c for c in state.frames if c not in ui.hotspot_cities(state))
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown('<div class="vs-section">1 · Add a photo</div>', unsafe_allow_html=True)
        up = st.file_uploader("Photo of smoke, dust, burning or haze (JPG/PNG)", type=["jpg", "jpeg", "png"])
        samples = {"Smoke from open burning": "smoke_plume.jpg", "Industrial stack": "industrial_stack.jpg",
                   "Construction dust": "dust_site.jpg", "Hazy traffic": "hazy_traffic.jpg",
                   "Smoggy skyline": "hazy_skyline.jpg", "Clear day": "clear_skyline.jpg"}
        sample = st.selectbox("...or try an illustrative sample", list(samples), disabled=up is not None)
        img_bytes = up.getvalue() if up is not None else (config.SAMPLES_DIR / samples[sample]).read_bytes()
        img_name = up.name if up is not None else samples[sample]

        st.markdown('<div class="vs-section">2 · Where is it?</div>', unsafe_allow_html=True)
        city = st.selectbox("City", cities, index=cities.index(ss["city"]) if ss.get("city") in cities else 0, key="rep_city")
        analysis = vision.analyze_image(img_bytes)
        exif = analysis["exif"]
        wards = citizen.load_wards(city)
        if exif["lat"] is not None:
            lat, lon = exif["lat"], exif["lon"]
            st.success(f"Location read from photo GPS: {lat:.5f}, {lon:.5f}")
        elif wards:
            names = sorted({f["properties"]["ward_name"] for f in wards["features"]})
            default = names.index("Gharoli") if "Gharoli" in names else 0
            wname = st.selectbox("Ward", names, index=default)
            p = next(f["properties"] for f in wards["features"] if f["properties"]["ward_name"] == wname)
            lat, lon = p["clat"], p["clon"]
        else:
            c = config.CITIES[city]
            lat = st.number_input("Latitude", value=float(c.lat), format="%.5f")
            lon = st.number_input("Longitude", value=float(c.lon), format="%.5f")
        note = st.text_input("What do you see? (optional)", placeholder="e.g. garbage being burnt near the market")

    with right:
        st.markdown('<div class="vs-section">3 · Automatic analysis</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.image(Image.open(io.BytesIO(img_bytes)), caption=img_name, width="stretch")
        haze = 1.0 - analysis["transmission_map"].astype(np.float32) / 255.0      # 0 clear .. 1 fully veiled
        lo, hi = np.array([252, 252, 251], np.float32), np.array([166, 54, 3], np.float32)  # one-hue ramp
        heat = (lo[None, None, :] * (1 - haze[..., None]) + hi[None, None, :] * haze[..., None]).astype(np.uint8)
        b.image(heat, caption="Haze map (dark channel prior): darker orange = more veiled by haze/smoke", width="stretch")
        res = ui.get_hotspots(state.mode, now_str, city)
        sens = res["sensors"]
        dx, dy = km_offsets(lat, lon, sens["lat"].values, sens["lon"].values)
        near = sens[(np.hypot(dx, dy) <= 1.5) & (sens["flag"] == "ok")]
        excess = float(near["excess"].mean()) if len(near) and near["excess"].notna().any() else None
        taken = exif["taken"]
        mins_old = (pd.Timestamp.now() - pd.Timestamp(taken)).total_seconds() / 60 if taken is not None and state.mode == "live" else None
        cred, reasons = vision.report_credibility(analysis, 0.6, excess, mins_old, exif["lat"] is not None)
        cat_opts = vision.CATEGORIES
        k1, k2, k3 = st.columns(3)
        k1.markdown(ui.kpi("AI suggests", analysis["category"].split(" / ")[0], f"{analysis['category_confidence']:.0%} confidence"), unsafe_allow_html=True)
        k2.markdown(ui.kpi("Haze index", f"{analysis['haze_index']:.2f}", analysis["visibility"]), unsafe_allow_html=True)
        k3.markdown(ui.kpi("Report credibility", f"{cred:.0%}", "auto-verified evidence score"), unsafe_allow_html=True)
        st.markdown("<ul>" + "".join(f"<li>{ui.esc(r)}</li>" for r in reasons) + "</ul>", unsafe_allow_html=True)
        category = st.selectbox("Confirm or correct the category", cat_opts, index=cat_opts.index(analysis["category"]))
        st.caption("Indicative evidence only - the photo is fused with sensor data; it is not a PM2.5 measurement.")
        if st.button("Submit report", type="primary", icon="📨"):
            kind = vision.KIND_OF_CATEGORY.get(category, "haze")
            if kind == "clear":
                st.warning("A clear-sky photo was not submitted as a pollution report.")
            else:
                ss["user_reports"].append({
                    "report_id": f"U-{uuid.uuid4().hex[:6].upper()}", "time": str(state.now), "lat": float(lat),
                    "lon": float(lon), "category": category, "kind": kind, "haze_score": analysis["haze_index"],
                    "reporter_trust": 0.6, "image": img_name, "note": note or "Citizen photo report",
                    "source": "citizen (you)", "event_id": None, "city": city})
                st.success(f"Report submitted. It now feeds hotspot detection for {city} as evidence.")
                st.page_link("views/hotspot_radar.py", label=f"See it on the Hotspot Radar", icon="📍")

    st.markdown('<div class="vs-section">Recent reports</div>', unsafe_allow_html=True)
    res = ui.get_hotspots(state.mode, now_str, city)
    rep = res["reports"]
    rep = rep[(rep["time"] <= state.now + pd.Timedelta(minutes=59)) & (rep["time"] > state.now - pd.Timedelta(hours=24))]
    rep = rep.sort_values("time", ascending=False)
    if len(rep):
        g = st.columns(4)
        for col, (_, r) in zip(g, rep.head(4).iterrows()):
            path = config.SAMPLES_DIR / str(r["image"])
            if path.exists():
                col.image(str(path), width="stretch")
            col.caption(f"**{r['category']}** · {pd.Timestamp(r['time']):%d %b %H:%M}  \n{r['note']} · {r['source']}")
        st.dataframe(rep[["time", "category", "haze_score", "reporter_trust", "note", "source"]].round({"haze_score": 2, "reporter_trust": 2}), hide_index=True)
    else:
        st.info("No reports in the last 24 hours.")

# ----------------------------------------------------------------------------
with tab2:
    city2 = st.selectbox("City", ui.hotspot_cities(state), key="net_city")
    res2 = ui.get_hotspots(state.mode, now_str, city2)
    sens = res2["sensors"]
    counts = sens["status"].value_counts()
    ui.kpi_row([
        ("Sensors", f"{len(sens)}", f"about 1 per ward in {city2}"),
        ("Multi-gas nodes (CO/NO2)", f"{int(sens['multigas'].sum())}", "fingerprint traffic vs combustion"),
        ("Online now", f"{len(sens) - int(counts.get('offline', 0))}", f"{counts.get('offline', 0)} offline"),
        ("Quarantined by QC", f"{int(counts.get('stuck', 0) + counts.get('suspect-drift', 0))}", "stuck / drifting"),
    ])
    a, b = st.columns(2, gap="medium")
    with a:
        hosts = sens["host"].value_counts().head(8)[::-1]
        fig = go.Figure(go.Bar(x=hosts.values, y=hosts.index, orientation="h", marker=dict(color=ui.SERIES[0], cornerradius=4),
                               width=0.55, text=hosts.values, textposition="outside", textfont=dict(color=ui.INK2),
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        ui.style_fig(fig, height=320, legend=False)
        fig.update_layout(title=dict(text="Who hosts the sensors", x=0, font=dict(size=13)))
        ui.show_fig(fig, key="hosts")
    with b:
        st.markdown('<div class="vs-section">Why calibration matters</div>', unsafe_allow_html=True)
        st.markdown("Low-cost optical sensors swell particle readings in humid air and differ unit to unit. VayuSetu "
                    "corrects every reading with a humidity growth factor and a co-location bias, then learns each "
                    "sensor's usual offset from its neighbours.")
        rd = res2["readings"]
        sid = sens[sens["status"] == "normal"]["sensor_id"].iloc[0]
        s = rd[(rd["sensor_id"] == sid) & (rd["time"] > state.now - pd.Timedelta(hours=48))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["time"], y=s["pm25_raw"], name="Raw reading", line=dict(color=ui.SERIES[3], width=2),
                                 hovertemplate="%{x|%d %b %H:%M}<br>Raw %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=s["time"], y=s["pm25_cal"], name="Calibrated", line=dict(color=ui.SERIES[0], width=2),
                                 hovertemplate="%{x|%d %b %H:%M}<br>Calibrated %{y:.0f}<extra></extra>"))
        fig.update_yaxes(title_text=f"PM2.5, sensor {sid}")
        ui.style_fig(fig, height=250)
        ui.show_fig(fig, key="cal")
        fig = go.Figure(go.Scatter(x=s["time"], y=s["rh"], name="Relative humidity", line=dict(color=ui.MUTED, width=2),
                                   hovertemplate="%{x|%d %b %H:%M}<br>RH %{y:.0f}%<extra></extra>"))
        fig.update_yaxes(title_text="Humidity (%)")
        ui.style_fig(fig, height=180, legend=False, margin=dict(l=10, r=10, t=10, b=10))
        ui.show_fig(fig, key="rh")
    with st.expander("View data: sensor registry"):
        st.dataframe(sens[["sensor_id", "ward_name", "host", "model", "status", "pm25", "baseline_ratio"]].round(2), hide_index=True)

# ----------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="vs-section">Build a VayuSetu citizen node (about ₹3,000)</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Parts:** Raspberry Pi (any model with USB) + **Nova SDS011** PM2.5/PM10 sensor (comes with a USB adapter). "
        "Optional: DHT22 humidity sensor.\n"
        "2. **Create a free ThingSpeak channel** with fields: 1 = PM2.5, 2 = PM10, 3 = temperature, 4 = humidity.\n"
        "3. **Run** `python hardware/sds011_node.py --port /dev/ttyUSB0 --write-key <YOUR_KEY>` on the Pi.\n"
        "4. **Add to Streamlit secrets:** `THINGSPEAK_CHANNEL_ID` (and `THINGSPEAK_READ_KEY` for private channels).\n"
        "5. The node appears here and in live mode. Place it 1.5-3 m high, away from direct exhaust and kitchen vents.")
    secrets = ui.get_secrets()
    ch = secrets.get("THINGSPEAK_CHANNEL_ID")
    if ch:
        r = live.fetch_thingspeak(ch, secrets.get("THINGSPEAK_READ_KEY"))
        if r.ok:
            d = r.df
            st.success(f"Live node '{d['name'].iloc[-1]}' - {len(d)} readings, latest {d['time'].iloc[-1]:%d %b %H:%M}: "
                       f"PM2.5 {d['pm25'].iloc[-1]:.0f} µg/m³")
            fig = go.Figure(go.Scatter(x=d["time"], y=d["pm25"], line=dict(color=ui.SERIES[0], width=2), name="PM2.5"))
            ui.style_fig(fig, height=260, legend=False)
            ui.show_fig(fig, key="ts_node")
        else:
            st.warning(f"ThingSpeak channel configured but not reachable: {r.message}")
    else:
        st.info("No hardware node configured yet - the network above is a simulated pilot deployment.")
    with st.expander("Node code (hardware/sds011_node.py)"):
        st.code((config.ROOT / "hardware" / "sds011_node.py").read_text(), language="python")
