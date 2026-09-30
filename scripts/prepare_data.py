"""
Build the bundled datasets used by VayuSetu.

Inputs (not stored in the repo, see README "Rebuilding the data"):
  RAW_DIR/city_hour.csv            CPCB hourly city data 2015-2020 (Kaggle
                                   "Air Quality Data in India", compiled from CPCB)
  GEO_RAW_DIR/*.geojson, *.shp     DataMeet ward / state boundaries

Outputs:
  data/cpcb/city_hour.parquet      cleaned hourly data (26 cities)
  data/geo/india_states.geojson    simplified Survey-of-India state boundaries
  data/geo/wards/<city>.geojson    simplified ward boundaries + centroids
  data/geo/sensors.json            citizen-sensor sites per city (pilot network design)

Usage:
  python scripts/prepare_data.py --raw-dir /path/to/raw --geo-raw-dir /path/to/geo
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vayusetu import config  # noqa: E402

POLLUTANTS = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "CO": "co", "SO2": "so2", "O3": "o3"}
BOUNDS = {"pm25": 1000, "pm10": 1500, "no2": 500, "co": 50, "so2": 500, "o3": 400}


# ----------------------------------------------------------------------------
# CPCB hourly data
# ----------------------------------------------------------------------------
def _flatline_mask(s: pd.Series, run: int = 8) -> pd.Series:
    """True where the value repeats unchanged for >= `run` consecutive hours."""
    same = s.diff().eq(0)
    grp = (~same).cumsum()
    run_len = same.groupby(grp).transform("size")
    return same & (run_len >= run - 1) & s.notna()


def _isolated_spike_mask(s: pd.Series) -> pd.Series:
    med = s.rolling(25, center=True, min_periods=6).median()
    prev, nxt = s.shift(1), s.shift(-1)
    return (s > 4 * med) & ((s - med) > 200) & (prev < 0.5 * s) & (nxt < 0.5 * s)


def build_cpcb(raw_csv: Path, out_parquet: Path) -> pd.DataFrame:
    print(f"Reading {raw_csv} ...")
    df = pd.read_csv(raw_csv, usecols=["City", "Datetime", *POLLUTANTS.keys()])
    df = df.rename(columns={"City": "city", "Datetime": "time", **POLLUTANTS})
    df["time"] = pd.to_datetime(df["time"])
    frames = []
    stats = {"negative_or_zero": 0, "out_of_bounds": 0, "flatline": 0, "spike": 0}
    for city, g in df.groupby("city", sort=True):
        g = g.drop_duplicates("time").set_index("time").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full)
        for col in POLLUTANTS.values():
            s = g[col].astype(float)
            bad = s <= 0
            stats["negative_or_zero"] += int(bad.sum())
            s = s.mask(bad)
            oob = s > BOUNDS[col]
            stats["out_of_bounds"] += int(oob.sum())
            s = s.mask(oob)
            fl = _flatline_mask(s)
            stats["flatline"] += int(fl.sum())
            s = s.mask(fl)
            if col in ("pm25", "pm10"):
                sp = _isolated_spike_mask(s)
                stats["spike"] += int(sp.sum())
                s = s.mask(sp)
            g[col] = s.astype("float32")
        g["city"] = city
        g.index.name = "time"
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["city"] = out["city"].astype("category")
    out = out[["city", "time", *POLLUTANTS.values()]]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"Wrote {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB), rows={len(out):,}")
    print("Cleaning stats:", stats)
    return out


# ----------------------------------------------------------------------------
# Geo
# ----------------------------------------------------------------------------
def _load_features(path: Path) -> list[dict]:
    txt = path.read_text(encoding="utf-8").strip()
    try:
        g = json.loads(txt)
        return g["features"] if "features" in g else [g]
    except json.JSONDecodeError:  # newline-delimited GeoJSON
        feats = []
        for line in txt.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            o = json.loads(line)
            feats.extend(o["features"] if o.get("type") == "FeatureCollection" else [o])
        return feats


WARD_SOURCES = {
    "Delhi": ("Delhi_Delhi_Wards.geojson", lambda p: (p.get("Ward_No"), p.get("Ward_Name"))),
    "Kochi": ("Kochi_KCH_wards.geojson", lambda p: (p.get("Ward_No"), p.get("Ward_Name"))),
    "Kolkata": ("Kolkata_kolkata.geojson", lambda p: (p.get("WARD"), f"Ward {p.get('WARD')}")),
    "Lucknow": ("Lucknow_Lucknow_ward_boundary.geojson", lambda p: (p.get("Ward Num"), p.get("Ward Name"))),
    "Bengaluru": ("Bangalore_BBMP.geojson", lambda p: (p.get("KGISWardNo"), p.get("KGISWardName"))),
    "Chennai": ("Chennai_Wards.geojson", lambda p: (p.get("Ward_No"), f"Ward {p.get('Ward_No')} ({str(p.get('Zone_Name', '')).title()})")),
    "Hyderabad": ("Hyderabad_ghmc-wards.geojson", lambda p: (str(p.get("name", "")).split(" ")[1] if str(p.get("name", "")).startswith("Ward ") else p.get("name"), p.get("name"))),
    "Ahmedabad": ("Ahmedabad_Wards.geojson", lambda p: (str(p.get("Name", "")).split(" ")[0], " ".join(str(p.get("Name", "")).split(" ")[1:]).title() or p.get("Name"))),
    "Jaipur": ("Jaipur_Jaipur_Wards.geojson", lambda p: (p.get("WARD_NO"), f"Ward {p.get('WARD_NO')} ({str(p.get('ZONE_NAME', '')).title()})")),
}

HOSTS = ["Government school", "Resident welfare association", "College campus", "Primary health centre",
         "Bus depot", "Panchayat / ward office", "Private school", "Shop-owner volunteer", "Metro station",
         "Anganwadi centre", "Temple / mosque / church trust", "Market association"]


def _round_coords(geom: dict, nd: int = 5) -> dict:
    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [rnd(x) for x in c]
    return {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}


def _area_km2(shape) -> float:
    lat = shape.centroid.y
    return shape.area * (111.32 ** 2) * math.cos(math.radians(lat))


def _random_point_in(shape, rng: np.random.Generator):
    from shapely.geometry import Point
    minx, miny, maxx, maxy = shape.bounds
    for _ in range(400):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if shape.contains(p):
            return p
    return shape.representative_point()


def build_india_states(geo_raw: Path, out: Path) -> None:
    import shapefile
    from shapely.geometry import mapping, shape as shp
    r = shapefile.Reader(str(geo_raw / "India-States"))
    feats = []
    for sr in r.shapeRecords():
        name = sr.record[0]
        geom = shp(sr.shape.__geo_interface__).buffer(0).simplify(0.02, preserve_topology=True)
        feats.append({"type": "Feature", "properties": {"name": name},
                      "geometry": _round_coords(mapping(geom), 3)})
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB), states={len(feats)}")


def build_wards(geo_raw: Path) -> dict[str, list]:
    from shapely.geometry import mapping, shape as shp
    shapes_by_city = {}
    config.WARDS_DIR.mkdir(parents=True, exist_ok=True)
    for city, (fname, parse) in WARD_SOURCES.items():
        feats = _load_features(geo_raw / fname)
        out_feats, shapes = [], []
        for i, f in enumerate(feats):
            if not f.get("geometry"):
                continue
            wid, wname = parse(f.get("properties") or {})
            wid = str(wid if wid not in (None, "") else i + 1)
            wname = str(wname if wname not in (None, "") else f"Ward {wid}").strip().title()
            g = shp(f["geometry"]).buffer(0)
            if g.is_empty:
                continue
            gs = g.simplify(0.00025, preserve_topology=True)
            rp = g.representative_point()
            area = _area_km2(g)
            out_feats.append({"type": "Feature",
                              "properties": {"ward_id": wid, "ward_name": wname,
                                             "clat": round(rp.y, 5), "clon": round(rp.x, 5),
                                             "area_km2": round(area, 3)},
                              "geometry": _round_coords(mapping(gs))})
            shapes.append((wid, wname, g, area))
        path = config.WARDS_DIR / f"{city.lower()}.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": out_feats}, separators=(",", ":")))
        print(f"Wrote {path.name}: wards={len(out_feats)} size={path.stat().st_size / 1e3:.0f} KB")
        shapes_by_city[city] = shapes
    return shapes_by_city


def _sensor_record(city_code: str, idx: int, lat: float, lon: float, ward_id, ward_name, rng) -> dict:
    multigas = bool(rng.random() < 0.3)
    return {
        "id": f"{city_code}-{idx:04d}",
        "lat": round(float(lat), 5), "lon": round(float(lon), 5),
        "ward_id": ward_id, "ward_name": ward_name,
        "host": HOSTS[int(rng.integers(len(HOSTS)))],
        "model": "PMS5003 + MiCS-6814 (CO/NO2)" if multigas else ("SDS011" if rng.random() < 0.5 else "PMS5003"),
        "multigas": multigas,
        "near_road": bool(rng.random() < 0.35),
        "bias": round(float(rng.lognormal(0, 0.08)), 3),  # unit-to-unit sensor bias
        "site_factor": round(float(rng.lognormal(0, 0.10)), 3),  # micro-environment
    }


def _land_mask(geo_raw: Path):
    """Prepared India land polygon (Survey of India states) to keep sensors off the sea."""
    import shapefile
    from shapely.geometry import shape as shp
    from shapely.ops import unary_union
    from shapely.prepared import prep
    r = shapefile.Reader(str(geo_raw / "India-States"))
    return prep(unary_union([shp(s.__geo_interface__).buffer(0) for s in r.shapes()]))


def build_sensors(shapes_by_city: dict[str, list], land=None) -> None:
    from shapely.geometry import Point
    all_sites = {}
    for name, city in config.CITIES.items():
        rng = np.random.default_rng(sum(map(ord, name)) * 7919)  # deterministic per city
        code = "".join(ch for ch in name.upper() if ch.isalpha())[:3]
        sensors = []
        if name in shapes_by_city:
            for wid, wname, g, area in shapes_by_city[name]:
                n = 1 if area < 6 else (2 if area < 20 else 3)
                if name == "Kochi" and area > 0.8:
                    n += 1
                for _ in range(n):
                    p = _random_point_in(g, rng)
                    sensors.append(_sensor_record(code, len(sensors) + 1, p.y, p.x, wid, wname, rng))
            # Priority siting: three extra sensors around each industrial area / landfill.
            for poi in config.POIS.get(name, []):
                if poi.kind not in ("industrial", "landfill", "port"):
                    continue
                for k in range(3):
                    r = rng.uniform(0.6, 1.6)
                    th = 2 * math.pi * (k / 3) + rng.uniform(-0.5, 0.5)
                    lat = poi.lat + r * math.sin(th) / 111.32
                    lon = poi.lon + r * math.cos(th) / (111.32 * math.cos(math.radians(poi.lat)))
                    pt = Point(lon, lat)
                    ward = next(((wid, wname) for wid, wname, g, _ in shapes_by_city[name] if g.contains(pt)), None)
                    if ward is None and (land is not None and not land.contains(pt)):
                        continue
                    wid, wname = ward if ward else (f"Near {poi.name}", f"Near {poi.name}")
                    rec = _sensor_record(code, len(sensors) + 1, lat, lon, wid, wname, rng)
                    rec["host"] = "Priority site (industrial / landfill cluster)"
                    sensors.append(rec)
        else:
            n = 70 if city.population_m > 5 else 45
            radius_km = 11 if city.population_m > 5 else 7
            tries = 0
            while len(sensors) < n and tries < n * 50:
                tries += 1
                r = radius_km * math.sqrt(rng.random())
                th = rng.uniform(0, 2 * math.pi)
                lat = city.lat + (r * math.sin(th)) / 111.32
                lon = city.lon + (r * math.cos(th)) / (111.32 * math.cos(math.radians(city.lat)))
                if land is not None and not land.contains(Point(lon, lat)):
                    continue  # skip points that fall in the sea
                zone = f"Zone {int((math.degrees(th) % 360) // 45) + 1}-{'C' if r < radius_km / 2 else 'O'}"
                sensors.append(_sensor_record(code, len(sensors) + 1, lat, lon, zone, zone, rng))
        all_sites[name.lower()] = sensors
    config.SENSORS_FILE.write_text(json.dumps(all_sites, separators=(",", ":")))
    print(f"Wrote {config.SENSORS_FILE.name} with sensor sites for {len(all_sites)} cities")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--geo-raw-dir", type=Path, required=True)
    ap.add_argument("--skip-cpcb", action="store_true")
    args = ap.parse_args()
    if not args.skip_cpcb:
        build_cpcb(args.raw_dir / "city_hour.csv", config.CPCB_PARQUET)
    build_india_states(args.geo_raw_dir, config.GEO_DIR / "india_states.geojson")
    shapes = build_wards(args.geo_raw_dir)
    build_sensors(shapes, _land_mask(args.geo_raw_dir))


if __name__ == "__main__":
    main()
