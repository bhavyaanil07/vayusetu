"""
Orchestration: builds the platform "state" for a moment in time and runs the
AI engines on it. Used by the Streamlit app, the REST API and the scripts.

Two modes:
  replay - the Delhi-NCR smog episode of Oct-Nov 2019 using real CPCB data
           (works offline; lets us compare forecasts with what really happened)
  live   - current Copernicus CAMS + weather + NASA FIRMS (+ CPCB if a key is set)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import aqi as A
from . import alerts as AL
from . import config
from .data import citizen, cpcb, live
from .models import forecaster as FC
from .models import hotspots as HS

SOURCE_KIND_BADGE = {"real": "REAL DATA", "model": "MODEL", "simulated": "SIMULATED", "modeled": "MODELED"}


@dataclass
class PlatformState:
    mode: str
    now: pd.Timestamp
    frames: dict                     # city -> hourly pollutant history (<= now)
    met: dict                        # city -> hourly weather (history + next 72 h)
    fires: pd.DataFrame
    snapshot: pd.DataFrame
    sources: dict = field(default_factory=dict)
    future_obs: dict = field(default_factory=dict)   # replay only: what actually happened next
    physics: dict = field(default_factory=dict)      # live only: CAMS forecast after now
    notices: list = field(default_factory=list)

    @property
    def exercise(self) -> bool:
        return self.mode == "replay"


@lru_cache(maxsize=1)
def load_forecaster() -> FC.Forecaster:
    return FC.Forecaster.load(config.MODELS_DIR / "forecaster.joblib")


def _snapshot(frames: dict, now: pd.Timestamp, source: str) -> pd.DataFrame:
    rows = []
    for city, g in frames.items():
        w = g.loc[now - pd.Timedelta(hours=23): now]
        if w["pm25"].notna().sum() == 0:
            continue
        last = w["pm25"].dropna()
        pm24 = float(w["pm25"].mean())
        pm10_24 = float(w["pm10"].mean()) if "pm10" in w and w["pm10"].notna().any() else np.nan
        a = A.aqi(pm25=pm24, pm10=pm10_24 if not np.isnan(pm10_24) else None)
        c = config.CITIES[city]
        rows.append({"city": city, "state": c.state, "lat": c.lat, "lon": c.lon,
                     "corridors": ", ".join(c.corridors), "pm25_now": float(last.iloc[-1]),
                     "pm25_24h": pm24, "pm10_24h": pm10_24, "aqi": float(a), "category": A.category(a),
                     "color": A.color_hex(a), "source": source})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# State builders
# ----------------------------------------------------------------------------
def build_replay_state(now=None) -> PlatformState:
    now = pd.Timestamp(now or config.REPLAY_DEFAULT_TIME).floor("h")
    frames, future = {}, {}
    for city in cpcb.cities_with_data():
        g = cpcb.city_frame(city)
        hist = g.loc[now - pd.Timedelta(days=14): now, ["pm25", "pm10", "no2", "co"]]
        if hist["pm25"].tail(72).notna().sum() < 6:
            continue
        frames[city] = hist
        future[city] = g.loc[now + pd.Timedelta(hours=1): now + pd.Timedelta(hours=72), ["pm25"]]
    met = {}
    for city, g in frames.items():
        times = pd.date_range(now - pd.Timedelta(days=8), now + pd.Timedelta(hours=72), freq="h")
        full = cpcb.city_frame(city)["pm25"].rolling(24, min_periods=6).mean()
        met[city] = citizen.modeled_met(city, times, full)
    fires = live.simulated_fires(now - pd.Timedelta(hours=24), now)
    sources = {
        "air_quality": {"label": "CPCB continuous monitoring stations (hourly, city average)", "kind": "real", "ok": True},
        "weather": {"label": "Climatology-based wind & humidity (replay)", "kind": "modeled", "ok": True},
        "fires": {"label": "Satellite-style fire detections, synthetic for replay (live mode: NASA FIRMS VIIRS)", "kind": "simulated", "ok": True},
        "citizen": {"label": "Citizen sensor network & photo reports (pilot simulation)", "kind": "simulated", "ok": True},
    }
    return PlatformState("replay", now, frames, met, fires, _snapshot(frames, now, "CPCB (2019 replay)"),
                         sources, future_obs=future,
                         notices=["Replay mode: real CPCB data from the Oct-Nov 2019 Delhi-NCR smog episode."])


def build_live_state(secrets: dict | None = None) -> PlatformState:
    secrets = secrets or {}
    aq = live.fetch_open_meteo_aq(past_days=10, forecast_days=3)
    if not aq.ok:
        st = build_replay_state()
        st.notices.insert(0, f"Live data unavailable ({aq.message}). Showing replay instead.")
        return st
    wx = live.fetch_open_meteo_weather(past_days=10, forecast_days=3)
    fires = live.fetch_firms(days=1, map_key=secrets.get("FIRMS_MAP_KEY"))
    cp = live.fetch_cpcb_live(secrets.get("DATA_GOV_IN_API_KEY"))
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
    ground = live.cpcb_city_means(cp.df) if cp.ok else {}
    frames, physics, met, notes = {}, {}, {}, []
    for city, g in aq.df.groupby("city"):
        g = g.set_index("time").sort_index()
        cols = [c for c in ["pm25", "pm10", "no2", "co"] if c in g]
        hist, fut = g.loc[:now, cols], g.loc[now + pd.Timedelta(hours=1):, ["pm25"]]
        if city in ground and hist["pm25"].notna().any():
            # nudge the 45-km model field towards the ground stations (simple bias correction)
            f = float(np.clip(ground[city] / max(float(hist["pm25"].dropna().iloc[-1]), 1.0), 0.5, 3.0))
            hist = hist.assign(pm25=hist["pm25"] * f, pm10=hist.get("pm10", hist["pm25"]) * f)
            fut = fut.assign(pm25=fut["pm25"] * f)
            notes.append(f"{city}: CAMS scaled x{f:.2f} to match CPCB stations")
        frames[city] = hist
        physics[city] = fut.reset_index().rename(columns={"index": "time"})
        if wx.ok:
            w = wx.df[wx.df["city"] == city].set_index("time").sort_index()
            met[city] = w[["wind_speed", "wind_dir", "rh", "temp", "precip"]]
        else:
            times = pd.date_range(now - pd.Timedelta(days=8), now + pd.Timedelta(hours=72), freq="h")
            met[city] = citizen.modeled_met(city, times, hist["pm25"].rolling(24, min_periods=6).mean())
    if not fires.ok:
        fires_df = pd.DataFrame(columns=["latitude", "longitude", "frp", "time", "satellite", "state"])
    else:
        fires_df = fires.df
    sources = {
        "air_quality": {"label": aq.source + (" + CPCB ground stations" if ground else ""), "kind": "real", "ok": True, "message": aq.message},
        "weather": {"label": wx.source, "kind": "real" if wx.ok else "modeled", "ok": wx.ok, "message": wx.message},
        "fires": {"label": fires.source, "kind": "real", "ok": fires.ok, "message": fires.message},
        "cpcb_live": {"label": cp.source, "kind": "real", "ok": cp.ok, "message": cp.message},
        "citizen": {"label": "Citizen sensor network & photo reports (pilot simulation)", "kind": "simulated", "ok": True},
    }
    return PlatformState("live", now, frames, met, fires_df, _snapshot(frames, now, aq.source), sources,
                         physics=physics, notices=notes)


def build_state(mode: str = "replay", now=None, secrets: dict | None = None) -> PlatformState:
    return build_live_state(secrets) if mode == "live" else build_replay_state(now)


# ----------------------------------------------------------------------------
# Engines on top of a state
# ----------------------------------------------------------------------------
def forecast_all(state: PlatformState, cities: list[str] | None = None) -> dict:
    model = load_forecaster()
    panel = model.neighbour_panel(state.frames, state.now)
    out = {}
    for city in cities or list(state.frames):
        try:
            fc = model.forecast(state.frames, city, state.now, panel=panel)
        except ValueError:
            continue
        if state.mode == "live" and city in state.physics and not state.physics[city].empty:
            fc = FC.hybrid_with_physics(fc, state.physics[city], w_ml=0.5)
        out[city] = fc
    return out


def wind_now(state: PlatformState, city: str) -> tuple[float, float, float]:
    m = state.met.get(city)
    if m is None or m.empty:
        return float("nan"), float("nan"), float("nan")
    row = m.loc[:state.now].iloc[-1] if len(m.loc[:state.now]) else m.iloc[0]
    return float(row["wind_speed"]), float(row["wind_dir"]), float(row["rh"])


def city_hotspots(state: PlatformState, city: str, extra_reports: pd.DataFrame | None = None) -> dict:
    now = state.now
    times = pd.date_range(now - pd.Timedelta(hours=167), now, freq="h")
    bg = state.frames[city]
    met = state.met[city]
    events = citizen.events_for(city, times[0], times[-1], replay=state.mode == "replay")
    readings = citizen.simulate_readings(city, times, bg, met, events)
    reports = citizen.simulate_reports(city, events, times[0], times[-1])
    if extra_reports is not None and len(extra_reports):
        er = extra_reports[extra_reports["city"] == city] if "city" in extra_reports else extra_reports
        if len(er):
            reports = pd.concat([reports, er.drop(columns=["city"], errors="ignore")], ignore_index=True)
    sites = citizen.load_sites(city)
    hs, table = HS.detect_hotspots(city, readings, sites, now, reports=reports, fires=state.fires)
    wards = citizen.load_wards(city)
    return {"city": city, "hotspots": hs, "sensors": table, "readings": readings, "reports": reports,
            "events": events, "ward_values": HS.ward_values(city, table, wards),
            "ward_excess": HS.ward_values(city, table, wards, column="excess"), "wards": wards}


def regional_transport(state: PlatformState, city: str) -> dict:
    ws, wd, _ = wind_now(state, city)
    up = HS.upwind_fires(city, wd, state.fires)
    up["wind_speed"], up["wind_dir"] = ws, wd
    return up


def build_alerts(state: PlatformState, forecasts: dict, hotspot_results: dict | None = None) -> list[AL.Alert]:
    alerts: list[AL.Alert] = []
    for city, fc in forecasts.items():
        alerts += AL.alerts_from_forecast(city, fc, state.now, exercise=state.exercise)
    for city in forecasts:
        if config.CITIES[city].lat < 24:
            continue
        up = regional_transport(state, city)
        a = AL.alert_from_regional(city, up, up["wind_speed"], up["wind_dir"], state.now, exercise=state.exercise,
                                   source_label="NASA VIIRS" if state.mode == "live" else "simulated replay")
        if a:
            alerts.append(a)
    for city, res in (hotspot_results or {}).items():
        alerts += AL.alerts_from_hotspots(city, res["hotspots"], state.now, exercise=state.exercise)
    return AL.sort_alerts(alerts)


def corridor_table(state: PlatformState, forecasts: dict) -> pd.DataFrame:
    rows = []
    snap = state.snapshot.set_index("city")
    for code in config.MAJOR_CORRIDORS:
        meta = config.CORRIDORS[code]
        for city in meta["path"]:
            if city not in snap.index:
                continue
            fc = forecasts.get(city)
            worst = max(fc["windows"], key=lambda w: w["aqi"]) if fc else None
            rows.append({
                "corridor": code, "corridor_name": meta["name"], "city": city,
                "aqi_now": snap.loc[city, "aqi"], "category_now": snap.loc[city, "category"],
                "aqi_max_72h": worst["aqi"] if worst else np.nan,
                "category_72h": worst["category"] if worst else "",
                "p_very_poor_72h": max(w["prob_very_poor"] for w in fc["windows"]) if fc else np.nan,
                "p_severe_72h": max(w["prob_severe"] for w in fc["windows"]) if fc else np.nan,
            })
    return pd.DataFrame(rows)
