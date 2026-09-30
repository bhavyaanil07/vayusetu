"""
Citizen-sourced data layer.

VayuSetu's hyper-local layer is a network of low-cost PM2.5 sensors hosted
by schools, resident welfare associations, clinics and shops (about one per
ward), plus geotagged photo reports from citizens.

No such network exists yet, so this module *simulates* one realistically so
the detection pipeline can be demonstrated and evaluated:

* background pollution comes from real data (CPCB in replay mode, Copernicus
  CAMS in live mode) with smooth spatial variation between neighbourhoods;
* hidden local events (industrial emissions, waste/landfill burning,
  construction dust, traffic congestion) add plumes that drift downwind;
* sensors have unit-to-unit bias, humidity artefacts (optical sensors read
  high in humid air), noise, dropouts and a few faulty units - the same
  problems a real network has, so quality control and calibration matter.

Every simulated record is labelled so the UI can show its provenance.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config

KAPPA_RH = 0.12  # hygroscopic growth coefficient for optical low-cost sensors


# ----------------------------------------------------------------------------
# Sensor sites
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _all_sites() -> dict:
    return json.loads(config.SENSORS_FILE.read_text())


@lru_cache(maxsize=32)
def load_sites(city: str) -> pd.DataFrame:
    return pd.DataFrame(_all_sites()[city.lower()])


@lru_cache(maxsize=32)
def load_wards(city: str) -> dict | None:
    path = config.WARDS_DIR / f"{city.lower()}.geojson"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def km_offsets(lat0: float, lon0: float, lat, lon):
    """Local flat-earth offsets (east, north) in km."""
    dx = (np.asarray(lon) - lon0) * 111.32 * math.cos(math.radians(lat0))
    dy = (np.asarray(lat) - lat0) * 111.32
    return dx, dy


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------
KIND_PROFILE = {
    # ratio = PM2.5/PM10 of the source, co/no2 = excess per ug/m3 PM2.5
    "industrial": {"ratio": 0.72, "co": 0.012, "no2": 0.25},
    "burning": {"ratio": 0.85, "co": 0.020, "no2": 0.10},
    "dust": {"ratio": 0.22, "co": 0.000, "no2": 0.02},
    "traffic": {"ratio": 0.60, "co": 0.015, "no2": 0.90},
}

LABELS = {
    "industrial": ["Unpermitted night-time furnace operation", "Stack emissions above consent limits",
                   "Plastic/rubber processing unit emissions"],
    "burning": ["Open waste burning", "Landfill fire (smouldering)", "Garden/leaf-litter burning",
                "Tyre and cable burning at scrap yard"],
    "dust": ["Uncovered construction site", "Road-dust resuspension on broken road",
             "Demolition debris without dust control"],
    "traffic": ["Severe congestion at junction", "Idling heavy vehicles at terminal",
                "Diesel bus queue at depot"],
}


@dataclass
class Event:
    id: str
    city: str
    kind: str
    lat: float
    lon: float
    start: pd.Timestamp
    end: pd.Timestamp
    peak: float           # PM2.5 excess at the source (ug/m3)
    scale_km: float       # plume length scale
    label: str
    poi: str | None = None
    daily_hours: tuple | None = None  # active hours of day for recurring events, e.g. (20, 4)

    def active_factor(self, times: pd.DatetimeIndex) -> np.ndarray:
        t = np.asarray(times.values)
        inside = (t >= np.datetime64(self.start)) & (t <= np.datetime64(self.end))
        f = inside.astype(float)
        if self.daily_hours is not None:
            a, b = self.daily_hours
            hr = times.hour.values
            on = ((hr >= a) | (hr < b)) if a > b else ((hr >= a) & (hr < b))
            f = f * on
        # soft ramp at the start (first two hours) so onsets look physical
        ramp = np.clip((t - np.datetime64(self.start)) / np.timedelta64(1, "h") / 2.0 + 0.5, 0.3, 1.0)
        return f * ramp

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = str(self.start), str(self.end)
        return d


def _nearby(lat, lon, rng, km=0.7):
    r = km * math.sqrt(rng.random())
    th = rng.uniform(0, 2 * math.pi)
    return (lat + r * math.sin(th) / 111.32,
            lon + r * math.cos(th) / (111.32 * math.cos(math.radians(lat))))


def procedural_events(city: str, day: pd.Timestamp, seed_extra: int = 0) -> list[Event]:
    """Random but reproducible hidden events for one city and day."""
    day = pd.Timestamp(day).normalize()
    seed = (int(day.strftime("%Y%m%d")) * 31 + sum(map(ord, city)) + seed_extra) % (2 ** 32)
    rng = np.random.default_rng(seed)
    sites = load_sites(city)
    pois = config.POIS.get(city, [])
    events = []
    n = int(rng.integers(3, 6))
    kinds = rng.choice(["industrial", "burning", "dust", "traffic"], size=n, p=[0.28, 0.32, 0.22, 0.18])
    for i, kind in enumerate(kinds):
        cand = [p for p in pois if p.kind == ("landfill" if kind == "burning" and rng.random() < 0.4 else kind)
                or (kind == "industrial" and p.kind == "port")]
        if cand and kind != "dust" and rng.random() < 0.8:
            p = cand[int(rng.integers(len(cand)))]
            lat, lon = _nearby(p.lat, p.lon, rng, 0.8)
            poi = p.name
        else:
            s = sites.iloc[int(rng.integers(len(sites)))]
            lat, lon = _nearby(s["lat"], s["lon"], rng, 1.0)
            poi = None
        if kind == "industrial":
            start, dur, hours = day + pd.Timedelta(hours=20), 9, (20, 5)
        elif kind == "burning":
            h0 = int(rng.integers(17, 21))
            start, dur, hours = day + pd.Timedelta(hours=h0), int(rng.integers(4, 7)), None
        elif kind == "dust":
            start, dur, hours = day + pd.Timedelta(hours=int(rng.integers(8, 11))), int(rng.integers(6, 9)), None
        else:
            start, dur, hours = day + pd.Timedelta(hours=17), 5, None
        peak = float(rng.uniform(70, 170)) * (1.4 if kind == "burning" else 1.0)
        events.append(Event(
            id=f"{city[:3].upper()}-{day:%m%d}-{i + 1}", city=city, kind=kind, lat=float(lat), lon=float(lon),
            start=start, end=start + pd.Timedelta(hours=dur), peak=peak,
            scale_km=float(rng.uniform(1.2, 2.2)), label=str(rng.choice(LABELS[kind])), poi=poi,
            daily_hours=hours))
    return events


def scenario_events_delhi() -> list[Event]:
    """Hand-authored hidden events for the Delhi Oct-Nov 2019 replay (simulated)."""
    T = pd.Timestamp
    poi = {p.name: p for p in config.POIS["Delhi"]}

    def at(name, dlat=0.0, dlon=0.0):
        p = poi[name]
        return p.lat + dlat, p.lon + dlon

    ev = [
        ("burning", *at("Ghazipur Landfill"), T("2019-10-29 11:00"), T("2019-10-31 07:00"), 190, 1.9,
         "Landfill fire (smouldering)", "Ghazipur Landfill", None),
        ("industrial", *at("Bawana Industrial Area", 0.004, 0.006), T("2019-10-26 20:00"), T("2019-11-06 05:00"), 150, 2.0,
         "Unpermitted night-time furnace operation", "Bawana Industrial Area", (20, 5)),
        ("industrial", *at("Wazirpur Industrial Area"), T("2019-10-28 20:00"), T("2019-11-04 05:00"), 120, 1.6,
         "Stack emissions above consent limits", "Wazirpur Industrial Area", (21, 4)),
        ("burning", 28.6792, 77.0668, T("2019-10-30 18:00"), T("2019-10-30 23:59"), 170, 1.5,
         "Open waste and cable burning at scrap yard", None, None),
        ("traffic", *at("Anand Vihar ISBT"), T("2019-10-24 00:00"), T("2019-11-17 23:00"), 75, 1.3,
         "Idling buses and heavy congestion", "Anand Vihar ISBT", (17, 22)),
        ("dust", 28.5921, 77.0460, T("2019-10-30 08:00"), T("2019-10-30 17:00"), 140, 1.4,
         "Uncovered construction site (Dwarka)", None, None),
        ("dust", 28.5500, 77.2500, T("2019-10-31 09:00"), T("2019-10-31 16:00"), 120, 1.3,
         "Demolition debris without dust control", None, None),
        ("burning", *at("Bhalswa Landfill"), T("2019-11-02 14:00"), T("2019-11-03 10:00"), 170, 1.8,
         "Landfill fire (smouldering)", "Bhalswa Landfill", None),
    ]
    out = []
    for i, (kind, lat, lon, s, e, peak, L, label, p, hours) in enumerate(ev):
        out.append(Event(f"DEL-S{i + 1:02d}", "Delhi", kind, lat, lon, s, e, float(peak), float(L), label, p, hours))
    return out


def events_for(city: str, start, end, replay: bool) -> list[Event]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    evs: list[Event] = []
    if replay and city == "Delhi":
        evs.extend(e for e in scenario_events_delhi() if e.end >= start and e.start <= end)
    for day in pd.date_range(start.normalize() - pd.Timedelta(days=1), end.normalize(), freq="D"):
        for e in procedural_events(city, day, seed_extra=0 if not replay else 7):
            if e.end >= start and e.start <= end:
                if replay and city == "Delhi" and e.kind in ("industrial",):
                    continue  # scenario already covers industrial clusters
                evs.append(e)
    return evs


# ----------------------------------------------------------------------------
# Meteorology for simulation
# ----------------------------------------------------------------------------
def modeled_met(city: str, times: pd.DatetimeIndex, pm25_24h: pd.Series | None = None) -> pd.DataFrame:
    """Climatology-style hourly wind and humidity used when observed weather is unavailable (replay).

    Stagnant (low-wind) conditions are paired with high pollution, as observed in north-Indian winters.
    """
    rng = np.random.default_rng(sum(map(ord, city)) + int(times[0].strftime("%Y%m%d")))
    n = len(times)
    hour = times.hour.values
    lat = config.CITIES[city].lat if city in config.CITIES else 20
    base_dir = 305.0 if lat > 24 else (240.0 if lat < 15 else 270.0)
    daily = np.repeat(rng.normal(0, 18, n // 24 + 2), 24)[:n]
    wdir = (base_dir + daily + 10 * np.sin(2 * np.pi * hour / 24)) % 360
    diurnal = 0.6 + 0.9 * np.clip(np.sin(np.pi * (hour - 8) / 10), 0, None)
    if pm25_24h is not None:
        stag = np.clip(1.6 - 0.0035 * pm25_24h.reindex(times).ffill().bfill().fillna(100).values, 0.35, 1.5)
    else:
        stag = 1.0
    ws = np.clip((1.2 + 1.8 * diurnal) * stag + rng.normal(0, 0.25, n), 0.3, 8.0)
    rh = np.clip(62 + 22 * np.cos(2 * np.pi * (hour - 5) / 24) + rng.normal(0, 4, n), 25, 97)
    return pd.DataFrame({"wind_speed": ws, "wind_dir": wdir, "rh": rh}, index=times)


# ----------------------------------------------------------------------------
# Readings
# ----------------------------------------------------------------------------
def _spatial_field(city: str, lat, lon) -> np.ndarray:
    rng = np.random.default_rng(sum(map(ord, city)) * 13)
    c = config.CITIES[city]
    x, y = km_offsets(c.lat, c.lon, lat, lon)
    f = np.zeros_like(np.asarray(x, dtype=float))
    for _ in range(5):
        wl = rng.uniform(6, 22)
        th = rng.uniform(0, np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        f += np.sin((x * np.cos(th) + y * np.sin(th)) * 2 * np.pi / wl + ph)
    return np.exp(0.10 * f / np.sqrt(5))


def plume_kernel(dx, dy, wind_dir_from, wind_speed, scale_km):
    """Relative concentration at offsets (dx east, dy north) from a source.

    Elongated downwind, narrower cross-wind; calmer winds -> rounder plume.
    """
    to = math.radians((wind_dir_from + 180.0) % 360.0)
    ux, uy = math.sin(to), math.cos(to)           # unit vector pointing downwind
    along = dx * ux + dy * uy
    cross = -dx * uy + dy * ux
    stretch = 1.0 + min(wind_speed, 6.0) / 3.0
    sig_y = scale_km * (0.55 + 0.1 * np.clip(along, 0, None))
    down = np.where(along >= 0, np.exp(-along / (scale_km * stretch)), np.exp(-(along ** 2) / (2 * (0.5 * scale_km) ** 2)))
    return down * np.exp(-(cross ** 2) / (2 * sig_y ** 2))


def simulate_readings(city: str, times: pd.DatetimeIndex, background: pd.DataFrame,
                      met: pd.DataFrame, events: list[Event], seed: int = 0) -> pd.DataFrame:
    """Simulate hourly sensor readings.

    background: hourly DataFrame with pm25 (+ optional pm10, no2, co) for the city (real data).
    met: hourly wind_speed (m/s), wind_dir (deg, from), rh (%).
    Returns long DataFrame: time, sensor_id, pm25_raw, pm10_raw, co, no2, rh, pm25_cal, pm10_cal, flag.
    """
    sites = load_sites(city)
    n_s, n_t = len(sites), len(times)
    rng = np.random.default_rng(seed + int(times[0].strftime("%Y%m%d%H")) % 100000 + sum(map(ord, city)))
    bg = background.reindex(times).interpolate(limit=6).ffill().bfill()
    pm25_bg = bg["pm25"].values.astype(float)
    pm10_bg = bg["pm10"].values.astype(float) if "pm10" in bg and bg["pm10"].notna().any() else pm25_bg / 0.55
    pm10_bg = np.where(np.isnan(pm10_bg), pm25_bg / 0.55, pm10_bg)
    no2_bg = bg["no2"].values.astype(float) if "no2" in bg and bg["no2"].notna().any() else np.full(n_t, 35.0)
    co_bg = bg["co"].values.astype(float) if "co" in bg and bg["co"].notna().any() else pm25_bg * 0.012
    no2_bg = np.where(np.isnan(no2_bg), 35.0, no2_bg)
    co_bg = np.where(np.isnan(co_bg), pm25_bg * 0.012, co_bg)

    field = _spatial_field(city, sites["lat"].values, sites["lon"].values)
    site_f = sites["site_factor"].values * field
    hour = times.hour.values
    rush = ((hour >= 8) & (hour <= 11)) | ((hour >= 17) & (hour <= 21))
    road = np.where(sites["near_road"].values[None, :], np.where(rush, 1.12, 1.03)[:, None], 1.0)

    true25 = pm25_bg[:, None] * site_f[None, :] * road
    true10 = pm10_bg[:, None] * site_f[None, :] * road
    no2 = no2_bg[:, None] * field[None, :] * np.where(sites["near_road"].values[None, :], np.where(rush, 1.35, 1.1)[:, None], 1.0)
    co = co_bg[:, None] * field[None, :]

    ws = met["wind_speed"].reindex(times).ffill().bfill().values
    wd = met["wind_dir"].reindex(times).ffill().bfill().values
    rh = met["rh"].reindex(times).ffill().bfill().values

    c0 = config.CITIES[city]
    sx, sy = km_offsets(c0.lat, c0.lon, sites["lat"].values, sites["lon"].values)
    for ev in events:
        act = ev.active_factor(times)
        if not act.any():
            continue
        ex, ey = km_offsets(c0.lat, c0.lon, ev.lat, ev.lon)
        dx, dy = sx - ex, sy - ey
        prof = KIND_PROFILE[ev.kind]
        for ti in np.nonzero(act)[0]:
            k = plume_kernel(dx, dy, wd[ti], ws[ti], ev.scale_km)
            dil = 2.0 / (0.5 + ws[ti])  # calm, stagnant air -> stronger local plumes
            exc = ev.peak * act[ti] * k * dil
            true25[ti] += exc
            true10[ti] += exc / prof["ratio"]
            co[ti] += exc * prof["co"]
            no2[ti] += exc * prof["no2"]

    # --- sensor artefacts
    bias = sites["bias"].values[None, :]
    hyg = 1.0 + KAPPA_RH * (np.clip(rh, 0, 96) / 100.0) ** 2 / (1.0 - np.clip(rh, 0, 96) / 100.0)
    noise = rng.lognormal(0, 0.07, size=(n_t, n_s))
    raw25 = true25 * bias * hyg[:, None] * noise + rng.normal(0, 2.5, size=(n_t, n_s))
    raw10 = true10 * bias * hyg[:, None] * rng.lognormal(0, 0.08, size=(n_t, n_s)) + rng.normal(0, 4, size=(n_t, n_s))
    raw25 = np.clip(raw25, 1, None)
    raw10 = np.maximum(np.clip(raw10, 1, None), raw25 * 1.02)

    flag = np.full((n_t, n_s), "ok", dtype=object)
    # dropouts
    drop = rng.random((n_t, n_s)) < 0.025
    # two faulty units per city: one stuck, one drifting high
    frng = np.random.default_rng(sum(map(ord, city)) * 101)
    faulty = frng.choice(n_s, size=min(2, n_s), replace=False)
    raw25[:, faulty[0]] = np.round(float(np.nanmedian(pm25_bg)) * 1.9, 1)
    if len(faulty) > 1:
        raw25[:, faulty[1]] = raw25[:, faulty[1]] * 2.6 + 40

    # calibration (co-location derived unit bias with small error + humidity correction)
    bias_est = bias * frng.lognormal(0, 0.03, size=bias.shape)
    cal25 = raw25 / (bias_est * hyg[:, None])
    cal10 = raw10 / (bias_est * hyg[:, None])
    raw25[drop] = np.nan
    raw10[drop] = np.nan
    cal25[drop] = np.nan
    cal10[drop] = np.nan

    multigas = sites["multigas"].values[None, :]
    co_obs = np.where(multigas, co * rng.lognormal(0, 0.1, size=(n_t, n_s)), np.nan)
    no2_obs = np.where(multigas, no2 * rng.lognormal(0, 0.1, size=(n_t, n_s)), np.nan)

    df = pd.DataFrame({
        "time": np.repeat(times.values, n_s),
        "sensor_id": np.tile(sites["id"].values, n_t),
        "pm25_raw": raw25.ravel().round(1), "pm10_raw": raw10.ravel().round(1),
        "pm25_cal": cal25.ravel().round(1), "pm10_cal": cal10.ravel().round(1),
        "co": co_obs.ravel().round(3), "no2": no2_obs.ravel().round(1),
        "rh": np.repeat(rh, n_s).round(0),
        "flag": flag.ravel(),
    })
    return df


# ----------------------------------------------------------------------------
# Photo reports
# ----------------------------------------------------------------------------
PHOTO_CATEGORIES = {
    "burning": "Smoke / open burning",
    "industrial": "Industrial smoke / stack",
    "dust": "Dust / construction",
    "traffic": "Vehicle exhaust / congestion",
    "haze": "Haze / smog",
}

SAMPLE_IMAGE = {"burning": "smoke_plume.jpg", "industrial": "industrial_stack.jpg",
                "dust": "dust_site.jpg", "traffic": "hazy_traffic.jpg", "haze": "hazy_skyline.jpg"}


def simulate_reports(city: str, events: list[Event], start, end, seed: int = 0) -> pd.DataFrame:
    """Geotagged photo reports that citizens would plausibly submit near events (plus some noise)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rng = np.random.default_rng(seed + sum(map(ord, city)) + int(start.strftime("%m%d%H")))
    prob = {"burning": 0.8, "industrial": 0.5, "dust": 0.6, "traffic": 0.35}
    rows = []
    for ev in events:
        if rng.random() > prob[ev.kind]:
            continue
        times = pd.date_range(max(ev.start, start), min(ev.end, end), freq="h")
        times = times[ev.active_factor(times) > 0]
        if len(times) == 0:
            continue
        for _ in range(int(rng.integers(1, 4))):
            t = min(times[int(rng.integers(len(times)))] + pd.Timedelta(minutes=int(rng.integers(0, 59))), end)
            lat, lon = _nearby(ev.lat, ev.lon, rng, 1.0)
            rows.append({"report_id": f"R-{ev.id}-{len(rows) + 1}", "time": t, "lat": lat, "lon": lon,
                         "category": PHOTO_CATEGORIES[ev.kind], "kind": ev.kind,
                         "haze_score": float(np.clip(rng.normal(0.62 if ev.kind != "dust" else 0.5, 0.12), 0.05, 0.98)),
                         "reporter_trust": float(np.clip(rng.normal(0.7, 0.15), 0.2, 0.99)),
                         "image": SAMPLE_IMAGE[ev.kind], "note": ev.label, "source": "simulated",
                         "event_id": ev.id})
    # background noise reports (generic haze complaints)
    sites = load_sites(city)
    n_noise = int(rng.integers(1, 4))
    for _ in range(n_noise):
        s = sites.iloc[int(rng.integers(len(sites)))]
        t = start + (end - start) * rng.random()
        rows.append({"report_id": f"R-{city[:3].upper()}-N{len(rows) + 1}", "time": t.floor("min"),
                     "lat": float(s["lat"]), "lon": float(s["lon"]), "category": PHOTO_CATEGORIES["haze"],
                     "kind": "haze", "haze_score": float(np.clip(rng.normal(0.45, 0.15), 0.05, 0.95)),
                     "reporter_trust": float(np.clip(rng.normal(0.55, 0.2), 0.1, 0.95)),
                     "image": SAMPLE_IMAGE["haze"], "note": "General haze complaint", "source": "simulated",
                     "event_id": None})
    return pd.DataFrame(rows)
