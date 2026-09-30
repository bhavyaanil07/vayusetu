"""Lightweight geometry helpers (no GIS dependencies needed at runtime)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from . import config

INDIA_BBOX = (6.0, 68.0, 37.5, 97.5)  # lat_min, lon_min, lat_max, lon_max


@lru_cache(maxsize=1)
def india_states() -> dict:
    return json.loads((config.GEO_DIR / "india_states.geojson").read_text())


def _rings(geom: dict):
    if geom["type"] == "Polygon":
        yield geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly


def _in_ring(lon: np.ndarray, lat: np.ndarray, ring) -> np.ndarray:
    """Vectorised even-odd ray casting."""
    r = np.asarray(ring, dtype=float)
    x1, y1 = r[:, 0], r[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    inside = np.zeros(len(lon), dtype=bool)
    for a, b, c, d in zip(x1, y1, x2, y2):
        cond = ((b > lat) != (d > lat)) & (lon < (c - a) * (lat - b) / ((d - b) + 1e-12) + a)
        inside ^= cond
    return inside


def points_in_geometry(lat, lon, geom: dict) -> np.ndarray:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    res = np.zeros(len(lat), dtype=bool)
    for poly in _rings(geom):
        outer = np.asarray(poly[0])
        bb = (lon >= outer[:, 0].min()) & (lon <= outer[:, 0].max()) & (lat >= outer[:, 1].min()) & (lat <= outer[:, 1].max())
        if not bb.any():
            continue
        idx = np.nonzero(bb)[0]
        inside = _in_ring(lon[idx], lat[idx], poly[0])
        for hole in poly[1:]:
            inside &= ~_in_ring(lon[idx], lat[idx], hole)
        res[idx] |= inside
    return res


def assign_states(lat, lon) -> np.ndarray:
    """Name of the Indian state containing each point ('' if outside India)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    out = np.full(len(lat), "", dtype=object)
    todo = np.ones(len(lat), dtype=bool)
    for f in india_states()["features"]:
        if not todo.any():
            break
        idx = np.nonzero(todo)[0]
        m = points_in_geometry(lat[idx], lon[idx], f["geometry"])
        out[idx[m]] = f["properties"]["name"]
        todo[idx[m]] = False
    return out
