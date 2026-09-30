"""Open Data & API: architecture, data sources, schemas, API and downloads."""
import json

import pandas as pd
import streamlit as st

from vayusetu import alerts as AL
from vayusetu import config, ui

mode, now_str = ui.current_key()
state = ui.get_state(mode, now_str)
fcs = ui.get_forecasts(mode, now_str)

ui.page_header("Open Data & API",
               "Built for interoperability: open schemas, CAP alerts, GeoJSON, model cards and a REST API so any "
               "city or state can plug in", state)

st.image(str(config.ROOT / "docs" / "architecture.png"), width="stretch",
         caption="VayuSetu architecture: every city/state runs a node on its own data; a hub aggregates models only.")

st.markdown('<div class="vs-section">Data sources</div>', unsafe_allow_html=True)
src = pd.DataFrame([
    {"Source": "CPCB CAAQMS (hourly, 26 cities, 2015-2020)", "Used for": "Training, backtests, replay",
     "Resolution": "Station / city average", "Access": "Bundled (Kaggle compilation of CPCB data)", "Kind": "Real"},
    {"Source": "CPCB real-time stations (data.gov.in)", "Used for": "Live ground truth, CAMS bias nudging",
     "Resolution": "Hundreds of CAAQMS stations, hourly", "Access": "Free API key (optional)", "Kind": "Real"},
    {"Source": "Copernicus CAMS via Open-Meteo", "Used for": "Live analysis + 3-day physics forecast",
     "Resolution": "0.4° (~45 km), hourly", "Access": "Free, no key", "Kind": "Real (model, satellite-assimilated)"},
    {"Source": "Open-Meteo weather", "Used for": "Wind, humidity (calibration, plume direction)",
     "Resolution": "~11 km, hourly", "Access": "Free, no key", "Kind": "Real"},
    {"Source": "NASA FIRMS VIIRS active fires", "Used for": "Crop/waste burning, upwind smoke",
     "Resolution": "375 m, ~2 passes/day", "Access": "Open files or free MAP_KEY", "Kind": "Real (satellite)"},
    {"Source": "NASA GIBS imagery", "Used for": "True-colour / aerosol map overlay", "Resolution": "250 m-1 km, daily",
     "Access": "Free WMS", "Kind": "Real (satellite)"},
    {"Source": "DataMeet ward & state boundaries", "Used for": "Ward maps, state attribution",
     "Resolution": "Ward polygons (9 cities)", "Access": "CC BY / CC BY-SA", "Kind": "Real"},
    {"Source": "Citizen sensors & photo reports", "Used for": "Hyper-local hotspot detection",
     "Resolution": "~1 sensor per ward", "Access": "Pilot design - simulated here", "Kind": "Simulated"},
])
st.dataframe(src, hide_index=True)

st.markdown('<div class="vs-section">Open schemas</div>', unsafe_allow_html=True)
tabs = st.tabs(["Observation", "Citizen report", "Hotspot", "Forecast", "Alert (CAP 1.2)"])
for tab, name in zip(tabs[:4], ["observation", "citizen_report", "hotspot", "forecast"]):
    with tab:
        schema = json.loads((config.ROOT / "docs" / "schemas" / f"{name}.schema.json").read_text())
        st.json(schema, expanded=False)
        st.download_button(f"Download {name}.schema.json", json.dumps(schema, indent=2), f"{name}.schema.json",
                           "application/json", key=f"sch_{name}", on_click="ignore")
with tabs[4]:
    alerts = ui.get_alerts(mode, now_str, include_hotspots=False)
    if alerts:
        st.code(AL.to_cap_xml(alerts[0]), language="xml")
    st.caption("OASIS Common Alerting Protocol 1.2 - the standard used by NDMA's SACHET integrated alert system, "
               "so VayuSetu alerts can be relayed over SMS, cell broadcast, TV and radio channels that already exist.")

st.markdown('<div class="vs-section">REST API</div>', unsafe_allow_html=True)
st.markdown("Run `uvicorn api.main:app --port 8000` and open `/docs` for the interactive OpenAPI page.")
st.dataframe(pd.DataFrame([
    {"Method": "GET", "Endpoint": "/v1/cities", "Returns": "Cities, corridors, capabilities"},
    {"Method": "GET", "Endpoint": "/v1/snapshot?mode=live", "Returns": "Current AQI for every city + data provenance"},
    {"Method": "GET", "Endpoint": "/v1/forecast/{city}", "Returns": "72-h hourly forecast + 3 windows with spike probabilities"},
    {"Method": "GET", "Endpoint": "/v1/hotspots/{city}", "Returns": "GeoJSON hotspots with source attribution & evidence"},
    {"Method": "GET", "Endpoint": "/v1/alerts?fmt=cap", "Returns": "Alerts as JSON or CAP 1.2 XML"},
    {"Method": "POST", "Endpoint": "/v1/observations", "Returns": "Ingest a sensor reading (validated against the schema)"},
    {"Method": "POST", "Endpoint": "/v1/reports", "Returns": "Ingest a citizen report"},
    {"Method": "GET", "Endpoint": "/v1/models, /v1/models/{id}/weights", "Returns": "Model cards and exchangeable weights"},
]), hide_index=True)

st.markdown('<div class="vs-section">Download the current picture</div>', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
d1.download_button("City snapshot (CSV)", state.snapshot.round(2).to_csv(index=False), "vayusetu_snapshot.csv", "text/csv",
                   on_click="ignore")
fc_json = {c: {"issue_time": str(f["issue_time"]),
               "windows": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in w.items()} for w in f["windows"]]}
           for c, f in fcs.items()}
d2.download_button("Forecasts (JSON)", json.dumps(fc_json, indent=2, default=str), "vayusetu_forecasts.json",
                   "application/json", on_click="ignore")
res = ui.get_hotspots(state.mode, now_str, "Delhi") if "Delhi" in state.frames else None
if res:
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
         "properties": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in h.items() if k not in ("lat", "lon")}}
        for h in res["hotspots"]]}
    d3.download_button("Delhi hotspots (GeoJSON)", json.dumps(gj, indent=2, default=str), "delhi_hotspots.geojson",
                       "application/geo+json", on_click="ignore")

with st.expander("Credits & licences"):
    st.markdown(
        "- **CPCB** air-quality data (Government of India); historical hourly compilation from the Kaggle dataset "
        "*Air Quality Data in India (2015-2020)*.\n"
        "- **Copernicus Atmosphere Monitoring Service** data via **Open-Meteo** (CC BY 4.0).\n"
        "- **NASA FIRMS** and **NASA GIBS** (open, NASA EOSDIS).\n"
        "- Boundaries: **DataMeet** India states (CC BY 4.0, Survey of India outlines) and municipal wards (CC BY-SA 2.5 IN).\n"
        "- Methods: dark channel prior (He, Sun & Tang, 2009), Getis-Ord Gi*, DBSCAN, FedAvg (McMahan et al., 2017), "
        "DP-FedAvg, conformal prediction, NAQI (CPCB, 2014), GRAP (CAQM).")
