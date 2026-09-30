"""
Hidden-hotspot detection and source attribution.

Pipeline (runs every hour per city):
 1. Quality control      - drop offline, stuck or drifting sensors.
 2. Local background     - robust (35th percentile) level of sensors within 6 km,
                           so a whole-city smog episode is not mistaken for a hotspot.
 3. Excess & spatial stats - excess over background, robust z-scores and the
                           Getis-Ord Gi* statistic (is a sensor surrounded by
                           other high-excess sensors?).
 4. Clustering           - DBSCAN groups hot sensors into hotspots.
 5. Evidence fusion      - citizen photos, satellite fire detections, known
                           industrial areas / landfills / junctions nearby.
 6. Source attribution   - an explainable points system using chemical
                           fingerprints (PM2.5/PM10 ratio, CO and NO2 excess),
                           time-of-day patterns and the evidence above.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .. import config
from ..data.citizen import km_offsets

TH = config.THRESHOLDS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def pairwise_km(lat, lon) -> np.ndarray:
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    x, y = km_offsets(lat0, lon0, lat, lon)
    return np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)


def getis_ord_gi_star(values: np.ndarray, dist_km: np.ndarray, band_km: float) -> np.ndarray:
    """Getis-Ord Gi* z-scores with binary distance-band weights (self included)."""
    n = len(values)
    if n < 3:
        return np.zeros(n)
    W = (dist_km <= band_km).astype(float)
    xbar = values.mean()
    s = math.sqrt(max((values ** 2).mean() - xbar ** 2, 1e-9))
    wsum = W.sum(1)
    w2 = (W ** 2).sum(1)
    num = W @ values - xbar * wsum
    den = s * np.sqrt(np.maximum((n * w2 - wsum ** 2) / (n - 1), 1e-9))
    return num / den


# ----------------------------------------------------------------------------
# Quality control
# ----------------------------------------------------------------------------
def quality_control(readings: pd.DataFrame, sites: pd.DataFrame, at_time) -> dict[str, str]:
    """Flag sensors as offline / stuck / suspect using the last 48 hours."""
    at_time = pd.Timestamp(at_time)
    r = readings[(readings["time"] > at_time - pd.Timedelta(hours=48)) & (readings["time"] <= at_time)]
    flags = {}
    if r.empty:
        return flags
    piv = r.pivot_table(index="time", columns="sensor_id", values="pm25_raw", aggfunc="mean")
    last3 = piv.tail(3)
    city_med = piv.median(axis=1)
    for sid in sites["id"]:
        if sid not in piv.columns or last3[sid].isna().all():
            flags[sid] = "offline"
            continue
        s = piv[sid].dropna()
        if len(s) >= 6 and s.tail(12).std() < 0.5:
            flags[sid] = "stuck"
            continue
        ratio = (piv[sid] / city_med).dropna()
        if len(ratio) >= 24 and (ratio > 1.8).mean() >= 0.92:
            flags[sid] = "suspect-drift"
    return flags


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def local_background(values: np.ndarray, dist_km: np.ndarray, radius_km: float = 6.0, q: float = 35) -> np.ndarray:
    bg = np.empty(len(values))
    city_q = np.nanpercentile(values, q)
    for i in range(len(values)):
        m = (dist_km[i] <= radius_km) & (dist_km[i] > 0) & ~np.isnan(values)
        bg[i] = np.nanpercentile(values[m], q) if m.sum() >= 5 else city_q
    return bg


def _nearest_poi(city: str, lat: float, lon: float) -> dict[str, tuple[float, str]]:
    out = {}
    for p in config.POIS.get(city, []):
        dx, dy = km_offsets(lat, lon, p.lat, p.lon)
        d = float(math.hypot(dx, dy))
        kind = "industrial" if p.kind in ("industrial", "port") else p.kind
        if kind not in out or d < out[kind][0]:
            out[kind] = (d, p.name)
    return out


def attribute_source(ratio: float, co_per_pm: float, no2_per_pm: float, night_frac: float,
                     evening_frac: float, day_frac: float, rush_frac: float,
                     poi: dict, photos: dict, fires_near: int) -> tuple[dict, list[str]]:
    """Explainable points system -> probability per source type + evidence sentences."""
    s = {k: 0.0 for k in config.SOURCE_TYPES}
    ev: list[tuple[float, str]] = []
    r = 0.55 if np.isnan(ratio) else float(np.clip(ratio, 0, 1.5))
    comb = float(_sigmoid((r - 0.52) / 0.07))
    dusty = float(_sigmoid((0.42 - r) / 0.06))
    s["industrial"] += 1.8 * comb
    s["burning"] += 2.0 * float(_sigmoid((r - 0.62) / 0.07))
    s["dust"] += 3.2 * dusty
    s["traffic"] += 1.0 * float(_sigmoid((r - 0.45) / 0.08)) * float(_sigmoid((0.8 - r) / 0.08))
    if not np.isnan(ratio):
        if r >= 0.6:
            ev.append((1.5, f"Fine-particle share PM2.5/PM10 = {r:.2f} → combustion signature"))
        elif r <= 0.42:
            ev.append((1.5, f"Coarse-particle dominated (PM2.5/PM10 = {r:.2f}) → dust signature"))
        else:
            ev.append((0.5, f"PM2.5/PM10 = {r:.2f} (mixed sources)"))
    s["industrial"] += 1.6 * night_frac
    s["burning"] += 1.2 * evening_frac + 0.4 * night_frac
    s["dust"] += 1.4 * day_frac - 1.6 * night_frac
    s["traffic"] += 1.8 * rush_frac - 0.6 * night_frac
    if night_frac > 0.5:
        ev.append((1.2, f"{night_frac:.0%} of the excess occurs at night (20:00-05:00)"))
    elif rush_frac > 0.5:
        ev.append((1.0, f"Peaks align with rush hours ({rush_frac:.0%} of excess)"))
    elif day_frac > 0.5:
        ev.append((1.0, f"Daytime activity pattern ({day_frac:.0%} of excess 08:00-17:00)"))
    elif evening_frac > 0.4:
        ev.append((1.0, f"Evening pattern ({evening_frac:.0%} of excess 17:00-23:00)"))
    if not np.isnan(no2_per_pm) and no2_per_pm > 0.45:
        s["traffic"] += 2.0
        ev.append((1.4, f"NO2 rises {no2_per_pm:.2f} µg/m³ per µg/m³ of PM2.5 → vehicle exhaust"))
    if not np.isnan(co_per_pm) and co_per_pm > 0.012:
        s["burning"] += 0.8
        s["industrial"] += 0.5
        ev.append((0.8, f"Carbon monoxide co-rises ({co_per_pm * 1000:.0f} µg/m³ CO per µg/m³ PM2.5) → combustion"))
    for kind, (d, name) in poi.items():
        if kind == "industrial" and d < 3:
            s["industrial"] += 2.2 if d < 1.5 else 1.0
            ev.append((1.3, f"{d:.1f} km from {name}"))
        elif kind == "landfill" and d < 3:
            s["burning"] += 2.2 if d < 1.5 else 0.9
            ev.append((1.3, f"{d:.1f} km from {name}"))
        elif kind == "traffic" and d < 2:
            s["traffic"] += 2.0 if d < 1.0 else 1.0
            ev.append((1.1, f"{d:.1f} km from {name}"))
    kind_map = {"burning": "burning", "industrial": "industrial", "dust": "dust", "traffic": "traffic"}
    for k, n in photos.items():
        if k in kind_map and n > 0:
            s[kind_map[k]] += 1.3 + 0.3 * min(n - 1, 3)
            ev.append((1.4, f"{n} citizen photo report(s): {config.SOURCE_TYPES[kind_map[k]]['label'].lower()}"))
    if fires_near > 0:
        s["burning"] += 2.2
        ev.append((1.8, f"{fires_near} satellite fire detection(s) within 2 km (NASA VIIRS)"))
    keys = list(s)
    z = np.array([s[k] for k in keys])
    p = np.exp(z - z.max())
    p = p / p.sum()
    probs = {k: float(v) for k, v in zip(keys, p)}
    ev.sort(key=lambda x: -x[0])
    return probs, [e[1] for e in ev]


def sensor_baselines(readings: pd.DataFrame, at_time, days: int = 7, q: float = 0.30) -> pd.Series:
    """Each sensor's usual level relative to the city (log-ratio, low percentile over `days`).

    Sensors in permanently dirtier micro-environments read higher every hour; learning that
    offset stops them raising false alarms. A low percentile is used so that episodic local
    events (a few hours a day) do not get absorbed into the baseline.
    """
    at_time = pd.Timestamp(at_time)
    h = readings[(readings["time"] > at_time - pd.Timedelta(days=days)) & (readings["time"] <= at_time)]
    piv = h.pivot_table(index="time", columns="sensor_id", values="pm25_cal", aggfunc="mean")
    if len(piv) < 24:
        return pd.Series(0.0, index=piv.columns)
    city_q = piv.quantile(0.35, axis=1)
    lr = np.log(piv.clip(lower=1)).sub(np.log(city_q.clip(lower=1)), axis=0)
    base = lr.quantile(q)
    return (base - base.median()).fillna(0.0)


def detect_hotspots(city: str, readings: pd.DataFrame, sites: pd.DataFrame, at_time,
                    window_h: int = 3, reports: pd.DataFrame | None = None,
                    fires: pd.DataFrame | None = None) -> tuple[list[dict], pd.DataFrame]:
    """Detect hotspots at `at_time`. Returns (hotspots, per-sensor table)."""
    at_time = pd.Timestamp(at_time)
    flags = quality_control(readings, sites, at_time)
    baseline = sensor_baselines(readings, at_time)
    win = readings[(readings["time"] > at_time - pd.Timedelta(hours=window_h)) & (readings["time"] <= at_time)]
    agg = win.groupby("sensor_id").agg(pm25=("pm25_cal", "mean"), pm10=("pm10_cal", "mean"),
                                       co=("co", "mean"), no2=("no2", "mean"), rh=("rh", "mean"))
    tab = sites.set_index("id").join(agg, how="left")
    tab["flag"] = [flags.get(i, "ok") for i in tab.index]
    tab["baseline_ratio"] = np.exp(baseline.reindex(tab.index).fillna(0.0).values)
    usable = (tab["flag"] == "ok") & tab["pm25"].notna()
    t = tab[usable].copy()
    lat, lon = t["lat"].values, t["lon"].values
    D = pairwise_km(lat, lon)
    adj = 1.0 / t["baseline_ratio"].values          # remove each sensor's usual offset
    x25 = t["pm25"].values * adj
    x10 = t["pm10"].values * adj
    t["bg_pm25"] = local_background(x25, D)
    t["bg_pm10"] = local_background(x10, D)
    t["excess"] = x25 - t["bg_pm25"].values
    t["rel_excess"] = t["excess"] / np.maximum(t["bg_pm25"], 1)
    mad = np.nanmedian(np.abs(t["excess"] - np.nanmedian(t["excess"]))) * 1.4826 + 5.0
    t["z"] = t["excess"] / mad
    t["gi_z"] = getis_ord_gi_star(np.nan_to_num(t["excess"].values), D, band_km=2.6)
    min_ex = np.maximum(TH.hotspot_min_excess, 0.12 * t["bg_pm25"].values)
    hot = (t["gi_z"] >= TH.hotspot_gi_z) & (t["excess"] >= min_ex) & (t["z"] >= TH.hotspot_resid_z)
    # strong single-sensor anomalies also count (sparse areas)
    hot |= (t["excess"] >= np.maximum(2.2 * TH.hotspot_min_excess, 0.28 * t["bg_pm25"].values)) & (t["z"] >= 3.8)
    t["hot"] = hot
    t["pm25_adj"] = x25
    t["pm10_adj"] = x10

    # hourly excess history for time-of-day patterns (last 24 h), baseline-adjusted
    hist = readings[(readings["time"] > at_time - pd.Timedelta(hours=24)) & (readings["time"] <= at_time)]
    piv = hist.pivot_table(index="time", columns="sensor_id", values="pm25_cal", aggfunc="mean")
    piv = piv.div(np.exp(baseline.reindex(piv.columns).fillna(0.0)), axis=1)
    ok_cols = [c for c in piv.columns if flags.get(c, "ok") == "ok"]
    hour_bg = piv[ok_cols].quantile(0.35, axis=1)

    hotspots: list[dict] = []
    hot_t = t[t["hot"]]
    if len(hot_t):
        c0 = config.CITIES[city]
        hx, hy = km_offsets(c0.lat, c0.lon, hot_t["lat"].values, hot_t["lon"].values)
        labels = DBSCAN(eps=TH.cluster_eps_km, min_samples=1).fit_predict(np.column_stack([hx, hy]))
        for lab in sorted(set(labels)):
            m = hot_t[labels == lab]
            w = np.clip(m["excess"].values, 1, None)
            clat, clon = float(np.average(m["lat"], weights=w)), float(np.average(m["lon"], weights=w))
            dx, dy = km_offsets(clat, clon, m["lat"].values, m["lon"].values)
            radius = float(np.max(np.hypot(dx, dy))) + 0.6
            peak_id = m["excess"].idxmax()
            d25 = float((m["pm25_adj"] - m["bg_pm25"]).clip(lower=0).sum())
            d10 = float((m["pm10_adj"] - m["bg_pm10"]).clip(lower=0).sum())
            ratio = d25 / d10 if d10 > 1 else float("nan")
            # gas fingerprints from multi-gas members vs. multi-gas background
            mg = t[t["multigas"] & t["co"].notna()]
            mg_bg = mg[~mg["hot"]]
            mm = m[m["multigas"] & m["co"].notna()]
            if len(mm) and len(mg_bg) >= 3:
                dpm = max(float((mm["pm25_adj"] - mm["bg_pm25"]).mean()), 1.0)
                co_pp = (float(mm["co"].mean()) - float(mg_bg["co"].quantile(0.35))) / dpm
                no2_pp = (float(mm["no2"].mean()) - float(mg_bg["no2"].quantile(0.35))) / dpm
            else:
                co_pp = no2_pp = float("nan")
            # time-of-day profile of excess at the peak sensor
            if peak_id in piv.columns:
                ex = (piv[peak_id] - hour_bg).clip(lower=0).fillna(0)
                hrs = ex.index.hour
                tot = ex.sum() + 1e-6
                night = float(ex[(hrs >= 20) | (hrs < 5)].sum() / tot)
                evening = float(ex[(hrs >= 17) & (hrs < 23)].sum() / tot)
                day = float(ex[(hrs >= 8) & (hrs < 17)].sum() / tot)
                rush = float(ex[((hrs >= 8) & (hrs < 11)) | ((hrs >= 17) & (hrs < 21))].sum() / tot)
                above = (ex >= TH.hotspot_min_excess).values[::-1]
                persist = int(np.argmin(above)) if not above.all() else len(above)
                started = ex.index[-persist] if persist > 0 else at_time
            else:
                night = evening = day = rush = 0.0
                persist, started = 1, at_time
            photos, photo_ids = {}, []
            if reports is not None and len(reports):
                rr = reports[(reports["time"] > at_time - pd.Timedelta(hours=6)) & (reports["time"] <= at_time + pd.Timedelta(minutes=59))]
                if len(rr):
                    rx, ry = km_offsets(clat, clon, rr["lat"].values, rr["lon"].values)
                    near = rr[np.hypot(rx, ry) <= max(1.5, radius)]
                    for k, n in near["kind"].value_counts().items():
                        photos[k] = int(n)
                    photo_ids = near["report_id"].tolist()
            fires_near = 0
            if fires is not None and len(fires):
                fx, fy = km_offsets(clat, clon, fires["latitude"].values, fires["longitude"].values)
                fires_near = int((np.hypot(fx, fy) <= 2.0).sum())
            poi = _nearest_poi(city, clat, clon)
            probs, evidence = attribute_source(ratio, co_pp, no2_pp, night, evening, day, rush, poi, photos, fires_near)
            src = max(probs, key=probs.get)
            gi = float(m["gi_z"].max())
            conf = float(_sigmoid(-3.2 + 0.55 * min(gi, 6) + 0.45 * min(len(m), 5) + 0.25 * min(persist, 6)
                                  + (0.9 if photos else 0) + (1.0 if fires_near else 0)))
            peak_ex = float(m["excess"].max())
            severity = "Critical" if peak_ex >= 150 else ("High" if peak_ex >= 60 else "Moderate")
            wards = sorted(set(m["ward_name"].astype(str)))
            hotspots.append({
                "hotspot_id": f"HS-{city[:3].upper()}-{at_time:%m%d%H}-{lab + 1}",
                "city": city, "time": at_time, "lat": clat, "lon": clon, "radius_km": radius,
                "n_sensors": int(len(m)), "sensor_ids": m.index.tolist(),
                "peak_pm25": float(m["pm25"].max()), "peak_excess": peak_ex,
                "mean_excess": float(m["excess"].mean()), "background_pm25": float(m["bg_pm25"].median()),
                "gi_z": gi, "pm_ratio": ratio, "co_per_pm": co_pp, "no2_per_pm": no2_pp,
                "persistence_h": persist, "started": started,
                "source": src, "source_label": config.SOURCE_TYPES[src]["label"],
                "source_probs": probs, "evidence": evidence, "confidence": conf, "severity": severity,
                "wards": wards, "photo_reports": photo_ids, "fires_near": fires_near,
                "nearest_poi": {k: {"km": round(v[0], 2), "name": v[1]} for k, v in poi.items()},
            })
    hotspots.sort(key=lambda h: (-h["confidence"] * h["peak_excess"]))

    out = tab.copy()
    for col in ("bg_pm25", "excess", "rel_excess", "z", "gi_z", "hot"):
        out[col] = t[col] if col in t else np.nan
    out["hot"] = out["hot"].astype("boolean").fillna(False).astype(bool)
    out["chronic"] = (out["baseline_ratio"] >= 1.25) & (out["flag"] == "ok")
    out["status"] = np.where(out["flag"] != "ok", out["flag"], np.where(out["hot"], "hotspot", "normal"))
    out.index.name = "sensor_id"
    return hotspots, out.reset_index()


def ward_values(city: str, sensor_table: pd.DataFrame, wards_geo: dict | None, power: float = 2.0,
                radius_km: float = 4.0, column: str = "pm25") -> dict[str, float]:
    """Inverse-distance-weighted value (PM2.5 or local excess) at each ward centroid from calibrated sensors."""
    if not wards_geo:
        return {}
    ok = sensor_table[(sensor_table["flag"] == "ok") & sensor_table[column].notna()]
    if ok.empty:
        return {}
    c0 = config.CITIES[city]
    sx, sy = km_offsets(c0.lat, c0.lon, ok["lat"].values, ok["lon"].values)
    vals = ok[column].values.astype(float)
    out = {}
    for f in wards_geo["features"]:
        p = f["properties"]
        wx, wy = km_offsets(c0.lat, c0.lon, p["clat"], p["clon"])
        d = np.hypot(sx - wx, sy - wy)
        m = d <= radius_km
        if m.sum() == 0:
            m = d <= d.min() + 1e-6
        w = 1.0 / np.maximum(d[m], 0.3) ** power
        out[p["ward_id"]] = float(np.sum(w * vals[m]) / np.sum(w))
    return out


# ----------------------------------------------------------------------------
# Regional transport: satellite fires upwind of a city
# ----------------------------------------------------------------------------
def bearing_deg(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(np.asarray(lon2) - lon1)
    x = np.sin(dlon) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def upwind_fires(city: str, wind_dir_from: float, fires: pd.DataFrame | None,
                 max_km: float | None = None, half_angle: float | None = None) -> dict:
    """Count satellite fire detections in the upwind sector of a city."""
    max_km = max_km or TH.fire_upwind_km
    half_angle = half_angle or TH.fire_upwind_half_angle
    res = {"count": 0, "frp_mw": 0.0, "nearest_km": None, "sector": (wind_dir_from - half_angle, wind_dir_from + half_angle)}
    if fires is None or fires.empty or wind_dir_from is None or np.isnan(wind_dir_from):
        return res
    c = config.CITIES[city]
    fx, fy = km_offsets(c.lat, c.lon, fires["latitude"].values, fires["longitude"].values)
    dist = np.hypot(fx, fy)
    brg = bearing_deg(c.lat, c.lon, fires["latitude"].values, fires["longitude"].values)
    diff = np.abs((brg - wind_dir_from + 180) % 360 - 180)
    m = (dist <= max_km) & (diff <= half_angle)
    res["count"] = int(m.sum())
    res["frp_mw"] = float(fires.loc[m, "frp"].sum()) if "frp" in fires and m.any() else 0.0
    res["nearest_km"] = float(dist[m].min()) if m.any() else None
    return res
