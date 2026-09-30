import numpy as np
import pandas as pd

from vayusetu import engine
from vayusetu.models.forecaster import exceed_prob, hybrid_with_physics


def _state():
    return engine.build_replay_state("2019-10-30 22:00")


def test_forecast_shapes_and_quantiles():
    st = _state()
    fcs = engine.forecast_all(st, ["Delhi", "Mumbai"])
    fc = fcs["Delhi"]
    h = fc["hourly"]
    assert len(h) == 72 and h["h"].tolist() == list(range(1, 73))
    assert (h["p10"] <= h["p50"]).all() and (h["p50"] <= h["p90"]).all()
    assert len(fc["windows"]) == 3
    for w in fc["windows"]:
        assert 0 <= w["prob_severe"] <= w["prob_very_poor"] <= w["prob_poor"] <= 1
    # Delhi on 30 Oct 2019 was Severe; Mumbai was not
    assert fcs["Delhi"]["windows"][0]["prob_very_poor"] > 0.9
    assert fcs["Mumbai"]["windows"][0]["prob_very_poor"] < 0.2


def test_exceed_prob_monotonic():
    p = [exceed_prob(50, 100, 200, t) for t in (60, 100, 150, 300)]
    assert all(a > b for a, b in zip(p, p[1:]))
    assert abs(exceed_prob(50, 100, 200, 100) - 0.5) < 0.01


def test_hybrid_with_physics_moves_towards_physics():
    st = _state()
    fc = engine.forecast_all(st, ["Delhi"])["Delhi"]
    phys = pd.DataFrame({"time": fc["hourly"]["time"], "pm25": fc["hourly"]["p50"] * 2})
    hy = hybrid_with_physics(fc, phys, w_ml=0.5)
    ratio = (1 + hy["hourly"]["p50"]) / (1 + fc["hourly"]["p50"])
    assert np.all(ratio > 1.3) and np.all(ratio < 1.5)  # geometric mean of x and 2x ~ 1.41x
    assert hy["hybrid"] is True
