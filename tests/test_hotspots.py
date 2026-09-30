import numpy as np
import pandas as pd

from vayusetu import config, engine
from vayusetu.data import citizen
from vayusetu.models import hotspots as H


def test_gi_star_flags_cluster():
    rng = np.random.default_rng(0)
    lat = 28.6 + rng.uniform(-0.1, 0.1, 200)
    lon = 77.2 + rng.uniform(-0.1, 0.1, 200)
    D = H.pairwise_km(lat, lon)
    vals = rng.normal(0, 5, 200)
    core = D[0] < 2.0
    vals[core] += 60
    z = H.getis_ord_gi_star(vals, D, 2.6)
    assert z[core].mean() > 3 and abs(z[~core].mean()) < 1.5


def test_delhi_scenario_detects_landfill_fire_as_burning():
    st = engine.build_replay_state("2019-10-30 22:00")
    res = engine.city_hotspots(st, "Delhi")
    ghazipur = next(p for p in config.POIS["Delhi"] if p.name == "Ghazipur Landfill")
    near = [h for h in res["hotspots"]
            if np.hypot(*citizen.km_offsets(ghazipur.lat, ghazipur.lon, h["lat"], h["lon"])) < 3]
    assert near, "landfill fire not detected"
    assert near[0]["source"] == "burning" and near[0]["confidence"] > 0.6
    # faulty units are quarantined, not reported as hotspots
    flagged = res["sensors"]["status"].isin(["stuck", "suspect-drift"]).sum()
    assert flagged >= 1


def test_upwind_sector():
    fires = pd.DataFrame({"latitude": [30.5, 26.0], "longitude": [75.5, 80.0], "frp": [10.0, 10.0]})
    up = H.upwind_fires("Delhi", 305.0, fires)  # NW wind: Punjab is upwind, UP is not
    assert up["count"] == 1
    assert H.upwind_fires("Delhi", 120.0, fires)["count"] == 0


def test_attribution_fingerprints():
    dust, _ = H.attribute_source(0.2, float("nan"), float("nan"), 0.0, 0.0, 0.9, 0.2, {}, {}, 0)
    assert max(dust, key=dust.get) == "dust"
    traffic, _ = H.attribute_source(0.6, 0.01, 0.9, 0.1, 0.6, 0.2, 0.9, {"traffic": (0.5, "ITO")}, {}, 0)
    assert max(traffic, key=traffic.get) == "traffic"
    burn, _ = H.attribute_source(0.85, 0.02, 0.1, 0.4, 0.6, 0.1, 0.3, {"landfill": (0.6, "X")}, {"burning": 2}, 1)
    assert max(burn, key=burn.get) == "burning"
