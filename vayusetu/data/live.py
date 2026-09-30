"""
Live data connectors.

* Copernicus CAMS air-quality analysis & forecast (0.4 deg, satellite-assimilated)
  and weather (wind, humidity, temperature) via the free Open-Meteo APIs - no key.
* NASA FIRMS VIIRS active-fire detections (satellite, ~375 m) - open 24/48 h files,
  or the FIRMS API if a MAP_KEY is configured.
* CPCB real-time station readings via data.gov.in (needs a free API key).
* ThingSpeak channel for a real VayuSetu citizen sensor node (optional).

Every function returns a FetchResult and never raises, so the app can fall
back to the replay dataset when a service is unreachable.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..geo import INDIA_BBOX, assign_states

OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WX = "https://api.open-meteo.com/v1/forecast"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"
FIRMS_OPEN = {
    "VIIRS S-NPP": "/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_{win}.csv",
    "VIIRS NOAA-20": "/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_Asia_{win}.csv",
    "VIIRS NOAA-21": "/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_South_Asia_{win}.csv",
}
FIRMS_API_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
DATAGOV_AQI = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
THINGSPEAK = "https://api.thingspeak.com/channels/{channel}/feeds.json"
USER_AGENT = "VayuSetu/1.0 (clean-air hackathon prototype)"

AQ_VARS = {"pm2_5": "pm25", "pm10": "pm10", "nitrogen_dioxide": "no2", "carbon_monoxide": "co",
           "sulphur_dioxide": "so2", "ozone": "o3", "dust": "dust", "aerosol_optical_depth": "aod"}
WX_VARS = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "wind_speed_10m": "wind_speed",
           "wind_direction_10m": "wind_dir", "precipitation": "precip", "surface_pressure": "pressure",
           "cloud_cover": "cloud"}


@dataclass
class FetchResult:
    df: pd.DataFrame
    ok: bool
    source: str
    message: str = ""
    fetched_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None))


def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get(url: str, params: dict | None = None, timeout: float = 15.0, retries: int = 1):
    s = _session()
    last = None
    for attempt in range(retries + 1):
        try:
            r = s.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.6 * (attempt + 1))
    raise last


# ----------------------------------------------------------------------------
# Open-Meteo (Copernicus CAMS + weather)
# ----------------------------------------------------------------------------
def parse_open_meteo(js, cities: list[str], mapping: dict) -> pd.DataFrame:
    """Parse a single- or multi-location Open-Meteo response into a long DataFrame."""
    items = js if isinstance(js, list) else [js]
    frames = []
    for city, item in zip(cities, items):
        hourly = item.get("hourly") or {}
        if "time" not in hourly:
            continue
        d = {"time": pd.to_datetime(hourly["time"])}
        for src, dst in mapping.items():
            if src in hourly:
                d[dst] = pd.to_numeric(pd.Series(hourly[src]), errors="coerce").values
        f = pd.DataFrame(d)
        f["city"] = city
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["city", "time", *mapping.values()])
    out = pd.concat(frames, ignore_index=True)
    if "co" in out:  # Open-Meteo reports CO in ug/m3; CPCB/NAQI uses mg/m3
        out["co"] = out["co"] / 1000.0
    return out


def fetch_open_meteo_aq(cities: list[str] | None = None, past_days: int = 7, forecast_days: int = 3) -> FetchResult:
    cities = cities or config.city_names()
    params = {
        "latitude": ",".join(f"{config.CITIES[c].lat:.4f}" for c in cities),
        "longitude": ",".join(f"{config.CITIES[c].lon:.4f}" for c in cities),
        "hourly": ",".join(AQ_VARS), "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata", "domains": "cams_global",
    }
    try:
        js = _get(OPEN_METEO_AQ, params).json()
        df = parse_open_meteo(js, cities, AQ_VARS)
        return FetchResult(df, not df.empty, "Copernicus CAMS via Open-Meteo", f"{df['city'].nunique()} cities")
    except Exception as exc:  # noqa: BLE001
        return FetchResult(pd.DataFrame(), False, "Copernicus CAMS via Open-Meteo", f"unavailable: {exc}")


def fetch_open_meteo_weather(cities: list[str] | None = None, past_days: int = 7, forecast_days: int = 3) -> FetchResult:
    cities = cities or config.city_names()
    params = {
        "latitude": ",".join(f"{config.CITIES[c].lat:.4f}" for c in cities),
        "longitude": ",".join(f"{config.CITIES[c].lon:.4f}" for c in cities),
        "hourly": ",".join(WX_VARS), "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata", "wind_speed_unit": "ms",
    }
    try:
        js = _get(OPEN_METEO_WX, params).json()
        df = parse_open_meteo(js, cities, WX_VARS)
        return FetchResult(df, not df.empty, "Open-Meteo weather (ECMWF/NCEP/IMD blend)", f"{df['city'].nunique()} cities")
    except Exception as exc:  # noqa: BLE001
        return FetchResult(pd.DataFrame(), False, "Open-Meteo weather", f"unavailable: {exc}")


# ----------------------------------------------------------------------------
# NASA FIRMS
# ----------------------------------------------------------------------------
def parse_firms_csv(text: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "latitude" not in df:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "time", "confidence", "satellite", "daynight"])
    lat0, lon0, lat1, lon1 = INDIA_BBOX
    df = df[(df["latitude"].between(lat0, lat1)) & (df["longitude"].between(lon0, lon1))].copy()
    hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
    t_utc = pd.to_datetime(df["acq_date"].astype(str) + " " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True)
    df["time"] = t_utc.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["satellite"] = label
    if "frp" not in df:
        df["frp"] = np.nan
    if "confidence" in df:
        df["confidence"] = df["confidence"].astype(str)
    keep = ["latitude", "longitude", "frp", "time", "confidence", "satellite", "daynight"]
    return df[[c for c in keep if c in df.columns]]


def fetch_firms(days: int = 1, map_key: str | None = None, india_only: bool = True) -> FetchResult:
    frames, errors = [], []
    if map_key:
        for src in FIRMS_API_SOURCES:
            try:
                url = f"{FIRMS_BASE}/api/country/csv/{map_key}/{src}/IND/{max(1, min(days, 10))}"
                frames.append(parse_firms_csv(_get(url, timeout=25).text, src.replace("_NRT", "").replace("_", " ")))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{src}: {exc}")
    if not frames:
        win = "24h" if days <= 1 else ("48h" if days == 2 else "7d")
        for label, path in FIRMS_OPEN.items():
            try:
                frames.append(parse_firms_csv(_get(FIRMS_BASE + path.format(win=win), timeout=25).text, label))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: {exc}")
    frames = [f for f in frames if len(f)]
    if not frames:
        return FetchResult(pd.DataFrame(), False, "NASA FIRMS VIIRS", "; ".join(errors) or "no detections")
    df = pd.concat(frames, ignore_index=True)
    df["state"] = assign_states(df["latitude"].values, df["longitude"].values)
    if india_only:
        df = df[df["state"] != ""]
    return FetchResult(df.reset_index(drop=True), True, "NASA FIRMS VIIRS (375 m)",
                       f"{len(df):,} detections" + (f" ({len(errors)} feed errors)" if errors else ""))


# ----------------------------------------------------------------------------
# CPCB real-time (data.gov.in)
# ----------------------------------------------------------------------------
_POLL = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "SO2": "so2", "CO": "co", "OZONE": "o3", "NH3": "nh3"}


def parse_cpcb_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    val_col = next((c for c in ("avg_value", "pollutant_avg") if c in df), None)
    if val_col is None or "pollutant_id" not in df:
        return pd.DataFrame()
    df["value"] = pd.to_numeric(df[val_col].replace({"NA": np.nan}), errors="coerce")
    df["pollutant"] = df["pollutant_id"].map(_POLL)
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["last_update"] = pd.to_datetime(df.get("last_update"), format="%d-%m-%Y %H:%M:%S", errors="coerce")
    wide = df.pivot_table(index=["state", "city", "station", "last_update", "latitude", "longitude"],
                          columns="pollutant", values="value", aggfunc="mean", dropna=False).reset_index()
    wide.columns.name = None
    return wide


def fetch_cpcb_live(api_key: str | None) -> FetchResult:
    if not api_key:
        return FetchResult(pd.DataFrame(), False, "CPCB real-time (data.gov.in)", "no API key configured")
    records, offset = [], 0
    try:
        while offset < 10000:
            js = _get(DATAGOV_AQI, {"api-key": api_key, "format": "json", "limit": 1000, "offset": offset}).json()
            batch = js.get("records", [])
            records += batch
            total = int(js.get("total", len(records)) or len(records))
            offset += 1000
            if not batch or offset >= total:
                break
        df = parse_cpcb_records(records)
        return FetchResult(df, not df.empty, "CPCB real-time (data.gov.in)", f"{len(df)} stations")
    except Exception as exc:  # noqa: BLE001
        return FetchResult(pd.DataFrame(), False, "CPCB real-time (data.gov.in)", f"unavailable: {exc}")


def cpcb_city_means(stations: pd.DataFrame) -> dict[str, float]:
    """Average live PM2.5 per VayuSetu city (stations within 25 km)."""
    if stations is None or stations.empty or "pm25" not in stations:
        return {}
    from .citizen import km_offsets
    out = {}
    for name, c in config.CITIES.items():
        s = stations.dropna(subset=["latitude", "longitude", "pm25"])
        dx, dy = km_offsets(c.lat, c.lon, s["latitude"].values, s["longitude"].values)
        near = s[np.hypot(dx, dy) <= 25]
        if len(near):
            out[name] = float(near["pm25"].mean())
    return out


# ----------------------------------------------------------------------------
# ThingSpeak (real citizen node)
# ----------------------------------------------------------------------------
def parse_thingspeak(js: dict) -> pd.DataFrame:
    feeds = js.get("feeds") or []
    if not feeds:
        return pd.DataFrame()
    df = pd.DataFrame(feeds)
    out = pd.DataFrame({"time": pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)})
    for fld, name in (("field1", "pm25"), ("field2", "pm10"), ("field3", "temp"), ("field4", "rh")):
        if fld in df:
            out[name] = pd.to_numeric(df[fld], errors="coerce")
    ch = js.get("channel") or {}
    out["lat"] = pd.to_numeric(ch.get("latitude"), errors="coerce") if ch.get("latitude") else np.nan
    out["lon"] = pd.to_numeric(ch.get("longitude"), errors="coerce") if ch.get("longitude") else np.nan
    out["name"] = ch.get("name", "VayuSetu node")
    return out


def fetch_thingspeak(channel_id: str | None, read_key: str | None = None, results: int = 240) -> FetchResult:
    if not channel_id:
        return FetchResult(pd.DataFrame(), False, "ThingSpeak citizen node", "no channel configured")
    params = {"results": results}
    if read_key:
        params["api_key"] = read_key
    try:
        df = parse_thingspeak(_get(THINGSPEAK.format(channel=channel_id), params).json())
        return FetchResult(df, not df.empty, "ThingSpeak citizen node", f"{len(df)} readings")
    except Exception as exc:  # noqa: BLE001
        return FetchResult(pd.DataFrame(), False, "ThingSpeak citizen node", f"unavailable: {exc}")


# ----------------------------------------------------------------------------
# Satellite imagery overlay (NASA GIBS WMS, loaded by the viewer's browser)
# ----------------------------------------------------------------------------
def gibs_wms_url(layer: str, date: str, bbox: tuple[float, float, float, float], width: int = 1024,
                 height: int = 1024, fmt: str = "image/jpeg") -> str:
    """bbox = (lon_min, lat_min, lon_max, lat_max). WMS 1.1.1 uses lon/lat axis order."""
    lon0, lat0, lon1, lat1 = bbox
    return ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
            f"&LAYERS={layer}&STYLES=&FORMAT={fmt}&TRANSPARENT=TRUE&SRS=EPSG:4326"
            f"&BBOX={lon0},{lat0},{lon1},{lat1}&WIDTH={width}&HEIGHT={height}&TIME={date}")


GIBS_LAYERS = {
    "True colour (MODIS Terra)": "MODIS_Terra_CorrectedReflectance_TrueColor",
    "True colour (VIIRS NOAA-20)": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
    "Aerosol optical depth (MODIS)": "MODIS_Combined_Value_Added_AOD",
}


# ----------------------------------------------------------------------------
# Replay: synthetic fire detections (no archive access in the bundled dataset)
# ----------------------------------------------------------------------------
PADDY_DISTRICTS = [
    # name, lat, lon, weight  (Punjab & Haryana crop-residue burning belt)
    ("Sangrur", 30.25, 75.84, 1.0), ("Bathinda", 30.21, 74.95, 0.8), ("Ferozepur", 30.93, 74.61, 0.9),
    ("Moga", 30.82, 75.17, 0.7), ("Ludhiana", 30.90, 75.85, 0.6), ("Patiala", 30.34, 76.39, 0.8),
    ("Barnala", 30.38, 75.55, 0.6), ("Mansa", 29.99, 75.40, 0.6), ("Muktsar", 30.47, 74.52, 0.6),
    ("Faridkot", 30.67, 74.76, 0.5), ("Tarn Taran", 31.45, 74.93, 0.8), ("Amritsar", 31.63, 74.87, 0.6),
    ("Jalandhar", 31.33, 75.58, 0.4), ("Kapurthala", 31.38, 75.38, 0.4), ("Fatehgarh Sahib", 30.65, 76.39, 0.4),
    ("Kaithal", 29.80, 76.40, 0.5), ("Karnal", 29.69, 76.99, 0.4), ("Fatehabad", 29.52, 75.45, 0.5),
    ("Jind", 29.32, 76.31, 0.4), ("Kurukshetra", 29.97, 76.88, 0.4), ("Sirsa", 29.53, 75.03, 0.4),
]


def simulated_fires(start, end, peak_day: str = "2019-11-03", peak_per_day: int = 2400, seed: int = 5) -> pd.DataFrame:
    """Synthetic VIIRS-like fire detections following the typical Oct-Nov crop-burning curve.

    Used only in replay mode (clearly labelled); live mode uses NASA FIRMS.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rng = np.random.default_rng(seed)
    names, lat, lon, w = zip(*PADDY_DISTRICTS)
    w = np.array(w) / np.sum(w)
    rows = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        k = (day - pd.Timestamp(peak_day)).days
        n = int(peak_per_day * np.exp(-(k / 9.0) ** 2) * rng.lognormal(0, 0.25))
        if n <= 0:
            continue
        idx = rng.choice(len(names), size=n, p=w)
        la = np.array(lat)[idx] + rng.normal(0, 0.18, n)
        lo = np.array(lon)[idx] + rng.normal(0, 0.22, n)
        day_pass = rng.random(n) < 0.85
        t = day + pd.to_timedelta(np.where(day_pass, 13.5, 1.5) + rng.normal(0, 0.4, n), unit="h")
        rows.append(pd.DataFrame({"latitude": la, "longitude": lo, "frp": rng.lognormal(2.0, 0.7, n),
                                  "time": t, "confidence": np.where(rng.random(n) < 0.7, "n", "h"),
                                  "satellite": "Simulated (replay)", "daynight": np.where(day_pass, "D", "N"),
                                  "district": np.array(names)[idx]}))
    if not rows:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "time", "confidence", "satellite", "daynight", "state"])
    df = pd.concat(rows, ignore_index=True)
    df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    haryana = {"Kaithal", "Karnal", "Fatehabad", "Jind", "Kurukshetra", "Sirsa"}
    df["state"] = np.where(df["district"].isin(haryana), "Haryana", "Punjab")
    return df.reset_index(drop=True)
