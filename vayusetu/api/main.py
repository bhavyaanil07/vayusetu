"""
VayuSetu open REST API (optional service for city/state integration).

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs   (interactive OpenAPI)

Every endpoint accepts ?mode=replay|live and, for replay, ?time=2019-10-30T22:00.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vayusetu import alerts as AL  # noqa: E402
from vayusetu import config, engine  # noqa: E402
from vayusetu.schemas import CitizenReport, Observation  # noqa: E402

app = FastAPI(title="VayuSetu API", version=config.APP_VERSION,
              description="Open data & model exchange for federated clean-air intelligence: city snapshots, "
                          "72-h forecasts, hyper-local hotspots (GeoJSON), alerts (JSON or CAP 1.2), sensor "
                          "ingestion and model cards.")

_OBS: list[dict] = []
_REPORTS: list[dict] = []


def clean(obj):
    """Make any nested structure strict-JSON safe (NaN/inf -> null, numpy/pandas -> Python)."""
    import math

    import numpy as np
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@lru_cache(maxsize=16)
def _state(mode: str, time: str | None):
    if mode not in ("replay", "live"):
        raise HTTPException(400, "mode must be 'replay' or 'live'")
    return engine.build_state(mode, time)


@lru_cache(maxsize=16)
def _forecasts(mode: str, time: str | None):
    return engine.forecast_all(_state(mode, time))


def _t(time: str | None) -> str | None:
    return str(pd.Timestamp(time).floor("h")) if time else None


@app.get("/")
def root():
    return {"name": "VayuSetu API", "version": config.APP_VERSION, "docs": "/docs",
            "endpoints": ["/v1/cities", "/v1/snapshot", "/v1/forecast/{city}", "/v1/hotspots/{city}",
                          "/v1/alerts", "/v1/observations", "/v1/reports", "/v1/models"]}


@app.get("/v1/cities")
def cities():
    return [{"name": c.name, "lat": c.lat, "lon": c.lon, "state": c.state, "corridors": list(c.corridors),
             "ward_map": c.has_wards, "ncr": c.ncr} for c in config.CITIES.values()]


@app.get("/v1/snapshot")
def snapshot(mode: str = "replay", time: str | None = Query(None, examples=["2019-10-30T22:00"])):
    st = _state(mode, _t(time))
    return clean({"mode": st.mode, "time": st.now.isoformat(), "sources": st.sources,
                  "cities": st.snapshot.round(2).to_dict(orient="records")})


@app.get("/v1/forecast/{city}")
def forecast(city: str, mode: str = "replay", time: str | None = None):
    fcs = _forecasts(mode, _t(time))
    if city not in fcs:
        raise HTTPException(404, f"No forecast for {city}")
    fc = fcs[city]
    hourly = fc["hourly"][["time", "h", "p10", "p50", "p90"]].copy()
    hourly[["p10", "p50", "p90"]] = hourly[["p10", "p50", "p90"]].round(2)
    hourly["time"] = hourly["time"].astype(str)
    wins = [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in w.items()} for w in fc["windows"]]
    return clean({"city": city, "issue_time": str(fc["issue_time"]), "model_id": "vayucast-pm25-v1.1",
                  "windows": wins, "hourly": hourly.to_dict(orient="records")})


@app.get("/v1/hotspots/{city}")
def hotspots(city: str, mode: str = "replay", time: str | None = None):
    st = _state(mode, _t(time))
    if city not in st.frames:
        raise HTTPException(404, f"No data for {city}")
    res = engine.city_hotspots(st, city)
    feats = []
    for h in res["hotspots"]:
        props = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in h.items() if k not in ("lat", "lon")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
                      "properties": props})
    return JSONResponse(clean({"type": "FeatureCollection", "features": feats,
                               "metadata": {"city": city, "time": str(st.now), "provenance": st.sources["citizen"]["label"]}}))


@app.get("/v1/alerts")
def alerts(mode: str = "replay", time: str | None = None, fmt: str = Query("json", pattern="^(json|cap)$"),
           city: str | None = None):
    st = _state(mode, _t(time))
    fcs = _forecasts(mode, _t(time))
    hs = {c: engine.city_hotspots(st, c) for c in [city] if c and c in st.frames} if city else {}
    items = engine.build_alerts(st, fcs, hs)
    if city:
        items = [a for a in items if a.city == city]
    if fmt == "cap":
        body = "<alerts>\n" + "\n".join(AL.to_cap_xml(a).replace('<?xml version="1.0" ?>', "") for a in items) + "</alerts>"
        return Response(body, media_type="application/xml")
    return clean([a.to_dict() for a in items])


@app.post("/v1/observations", status_code=202)
def post_observation(obs: Observation):
    _OBS.append(obs.model_dump(mode="json"))
    return {"accepted": True, "stored": len(_OBS)}


@app.post("/v1/reports", status_code=202)
def post_report(rep: CitizenReport):
    if rep.city not in config.CITIES:
        raise HTTPException(404, "Unknown city")
    _REPORTS.append(rep.model_dump(mode="json"))
    return {"accepted": True, "stored": len(_REPORTS)}


@app.get("/v1/models")
def models():
    import json
    out = []
    for f in ("forecaster_card.json", "federated_card.json"):
        p = config.MODELS_DIR / f
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


@app.get("/v1/models/{model_id}/weights")
def model_weights(model_id: str):
    if model_id != "vayusetu-fl-global-v1":
        raise HTTPException(404, "Only the federated global model weights are exchangeable via the API")
    return Response((config.MODELS_DIR / "federated_global.npz").read_bytes(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=federated_global.npz"})
