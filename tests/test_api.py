"""REST API smoke tests (replay mode, no network needed)."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
T = "2019-10-30T22:00"


def test_root_and_cities():
    assert client.get("/").status_code == 200
    r = client.get("/v1/cities")
    assert r.status_code == 200 and len(r.json()) == 26


def test_snapshot_and_forecast():
    r = client.get("/v1/snapshot", params={"time": T})
    assert r.status_code == 200
    assert any(c["city"] == "Delhi" for c in r.json()["cities"])
    f = client.get("/v1/forecast/Delhi", params={"time": T}).json()
    assert len(f["windows"]) == 3 and len(f["hourly"]) == 72
    w = f["windows"][0]
    assert w["pm25_p10"] <= w["pm25_p50"] <= w["pm25_p90"]


def test_hotspots_geojson():
    g = client.get("/v1/hotspots/Delhi", params={"time": T}).json()
    assert g["type"] == "FeatureCollection" and len(g["features"]) >= 1
    assert g["features"][0]["geometry"]["type"] == "Point"


def test_alerts_json_and_cap():
    a = client.get("/v1/alerts", params={"time": T}).json()
    assert len(a) >= 1 and {"level", "title", "authorities"} <= set(a[0])
    x = client.get("/v1/alerts", params={"time": T, "fmt": "cap"})
    assert x.status_code == 200 and "urn:oasis:names:tc:emergency:cap:1.2" in x.text


def test_ingest_validation():
    ok = client.post("/v1/observations", json={"sensor_id": "DEL-0001", "time": "2019-10-30T22:00:00+05:30",
                                               "lat": 28.6, "lon": 77.2, "pm25": 180.5, "rh": 70})
    assert ok.status_code == 202
    bad = client.post("/v1/observations", json={"sensor_id": "X", "time": "2019-10-30T22:00:00+05:30",
                                                "lat": 128.6, "lon": 77.2, "pm25": -5})
    assert bad.status_code == 422


def test_models():
    m = client.get("/v1/models").json()
    assert {c["model_id"] for c in m} >= {"vayucast-pm25-v1.1", "vayusetu-fl-global-v1"}
    w = client.get("/v1/models/vayusetu-fl-global-v1/weights")
    assert w.status_code == 200 and len(w.content) > 1000
