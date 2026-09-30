"""
VayuCast - probabilistic PM2.5 forecaster for 1-72 hours ahead.

Two gradient-boosted model families, each trained for the 10th, 50th and
90th percentile (quantile regression):

* Window models predict the 24-hour average PM2.5 for the next 0-24 h,
  24-48 h and 48-72 h. India's AQI uses 24-hour averages, so these drive
  the AQI category, spike probabilities and alerts.
* Hourly models (forecast horizon given as a feature) supply the shape of
  the hourly curve, which is then rescaled to match the window forecasts.

Both predict the *change* from the last 24-hour mean (in log space) and use:
recent pollutant history, a seasonal/diurnal climatology prior, calendar
effects (including Diwali) and pollution in neighbouring cities (regional
transport, e.g. crop-residue smoke moving along the Indo-Gangetic plain).
The median is blended with persistence, and interval widths are
conformally calibrated on held-out data so that "80 % interval" really
covers about 80 % of outcomes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import aqi as aqi_mod
from .. import config
from . import features as F

QUANTILES = (0.1, 0.5, 0.9)
Z90 = 1.2815515655446004
NEIGHBOUR_KM = 600.0
WINDOW_LABELS = ["Next 24 h", "24-48 h", "48-72 h"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(d))


def neighbours_of(city: str, candidates) -> list[tuple[str, float]]:
    me = config.CITIES[city]
    out = []
    for o in candidates:
        if o == city or o not in config.CITIES:
            continue
        oc = config.CITIES[o]
        d = haversine_km(me.lat, me.lon, oc.lat, oc.lon)
        if d < NEIGHBOUR_KM:
            out.append((o, 1.0 / max(d, 25.0)))
    return out


def exceed_prob(p10, p50, p90, threshold: float, shrink: float = 1.0):
    """P(value > threshold) from a log-normal fitted to the 10/50/90 % quantiles."""
    l10, l50, l90 = (np.log1p(np.asarray(v, dtype=float)) for v in (p10, p50, p90))
    sigma = np.maximum((l90 - l10) / (2 * Z90) * shrink, 0.04)
    z = (np.log1p(threshold) - l50) / sigma
    return 1.0 - ndtr(z)


def anomaly_panel(frames: dict[str, pd.DataFrame], climatology: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-city 24-h mean log-PM2.5 anomaly and its 24-h change, on a common hourly index."""
    cols = {}
    for c, g in frames.items():
        lm = np.log1p(g["pm25"].reindex(index)).rolling(24, min_periods=8).mean()
        cl = F.clim_lookup(climatology.get(c), index)
        cols[c] = pd.Series(lm.values - cl, index=index)
        cols[c + "__d24"] = lm - lm.shift(24)
    return pd.DataFrame(cols, index=index)


def neighbour_features(city: str, panel: pd.DataFrame) -> pd.DataFrame:
    nb = [(o, w) for o, w in neighbours_of(city, [c for c in panel.columns if "__" not in c])]
    if not nb:
        return pd.DataFrame({"nb_anom24": np.nan, "nb_d24": np.nan, "nb_n": 0.0}, index=panel.index)
    w = np.array([x[1] for x in nb])
    an = panel[[o for o, _ in nb]].values
    dd = panel[[o + "__d24" for o, _ in nb]].values

    def wmean(M):
        m = ~np.isnan(M)
        W = np.where(m, w, 0.0)
        s = W.sum(1)
        return np.where(s > 0, np.nansum(np.where(m, M, 0.0) * W, 1) / np.maximum(s, 1e-12), np.nan)

    return pd.DataFrame({"nb_anom24": wmean(an), "nb_d24": wmean(dd),
                         "nb_n": (~np.isnan(an)).sum(1).astype(float)}, index=panel.index)


def window_rows(feats: pd.DataFrame, issues: pd.DatetimeIndex, lat: float, lon: float,
                clim: pd.DataFrame | None) -> pd.DataFrame:
    """Rows for the three 24-h windows of each issue time (stacked, with a `win` column)."""
    parts = []
    for w in range(3):
        X = F.build_rows(feats, issues, np.full(len(issues), 24 * w + 12), lat, lon, clim)
        X["win"] = float(w)
        if clim is not None:
            X["clim_win"] = np.mean([F.clim_lookup(clim, issues + pd.Timedelta(hours=24 * w + k))
                                     for k in range(1, 25)], axis=0)
        else:
            X["clim_win"] = np.nan
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


def _base(X: pd.DataFrame) -> np.ndarray:
    b = X["lp_mean24"].to_numpy(dtype=float, copy=True)
    nb = np.isnan(b)
    b[nb] = X["lp_0"].to_numpy(dtype=float)[nb]
    return b


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
@dataclass
class Forecaster:
    hourly_params: dict = field(default_factory=lambda: dict(
        max_iter=350, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.08,
        n_iter_no_change=25, random_state=42))
    window_params: dict = field(default_factory=lambda: dict(
        max_iter=450, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=42))
    hourly_models: dict = field(default_factory=dict)
    window_models: dict = field(default_factory=dict)
    hourly_features: list = field(default_factory=list)
    window_features: list = field(default_factory=list)
    climatology: dict = field(default_factory=dict)
    blend_weight: float = 0.6               # weight of the ML median vs persistence
    conformal: dict = field(default_factory=dict)   # win -> (lower, upper) log-space widening
    daily_shrink: float = 1.0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- training
    def fit_hourly(self, X: pd.DataFrame, y_log: np.ndarray) -> None:
        self.hourly_features = list(X.columns)
        base = _base(X)
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.hourly_params)
            m.fit(X.values.astype(np.float32), y_log - base)
            self.hourly_models[q] = m

    def fit_window(self, X: pd.DataFrame, y_log: np.ndarray) -> None:
        self.window_features = list(X.columns)
        base = _base(X)
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.window_params)
            m.fit(X.values.astype(np.float32), y_log - base)
            self.window_models[q] = m

    # ----------------------------------------------------------- prediction
    def _predict(self, models: dict, cols: list, X: pd.DataFrame) -> np.ndarray:
        Xv = X[cols].values.astype(np.float32)
        base = _base(X)
        P = np.column_stack([models[q].predict(Xv) for q in QUANTILES]) + base[:, None]
        P.sort(axis=1)
        return P  # log1p space

    def predict_hourly_log(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self.hourly_models, self.hourly_features, X)

    def predict_window_log(self, X: pd.DataFrame, persist24: np.ndarray | None = None,
                           calibrated: bool = True) -> np.ndarray:
        P = self._predict(self.window_models, self.window_features, X)
        if persist24 is not None and self.blend_weight < 1:
            lp = np.log1p(np.asarray(persist24, dtype=float))
            ok = ~np.isnan(lp)
            shift = np.where(ok, (1 - self.blend_weight) * (lp - P[:, 1]), 0.0)
            P = P + shift[:, None]
        if calibrated and self.conformal:
            wins = X["win"].values.astype(int)
            lo = np.array([self.conformal.get(int(w), (0.0, 0.0))[0] for w in wins])
            hi = np.array([self.conformal.get(int(w), (0.0, 0.0))[1] for w in wins])
            P[:, 0] = P[:, 0] - lo
            P[:, 2] = P[:, 2] + hi
        return P

    # ------------------------------------------------------------ inference
    def neighbour_panel(self, frames: dict[str, pd.DataFrame], issue_time) -> pd.DataFrame:
        """Pre-compute the regional anomaly panel once when forecasting many cities."""
        t0 = pd.Timestamp(issue_time).floor("h")
        idx = pd.date_range(t0 - pd.Timedelta(hours=72), t0, freq="h")
        return anomaly_panel({c: g.loc[:t0].tail(24 * 5) for c, g in frames.items()}, self.climatology, idx)

    def forecast(self, frames: dict[str, pd.DataFrame], city: str, issue_time,
                 lat: float | None = None, lon: float | None = None, panel: pd.DataFrame | None = None) -> dict:
        """Forecast for `city` at `issue_time`.

        frames: city -> hourly DataFrame (DatetimeIndex, pm25 [+pm10, no2, co]).
                Neighbouring cities in `frames` feed the regional-transport features.
        Returns {"hourly": DataFrame(time,h,p10,p50,p90), "windows": [dict, ...]}.
        """
        t0 = pd.Timestamp(issue_time).floor("h")
        meta = config.CITIES.get(city)
        lat = lat if lat is not None else (meta.lat if meta else 0.0)
        lon = lon if lon is not None else (meta.lon if meta else 0.0)
        hist = frames[city].loc[:t0]
        if hist.empty or hist["pm25"].tail(72).notna().sum() < 6:
            raise ValueError(f"Not enough recent PM2.5 history for {city}")
        h = F.complete_hourly(hist.tail(24 * 10), end=t0)
        feats = F.issue_features(h)
        if panel is None:
            panel = self.neighbour_panel(frames, t0)
        nb = neighbour_features(city, panel).reindex(feats.index)
        feats = pd.concat([feats, nb], axis=1)
        clim = self.climatology.get(city)
        issues = pd.DatetimeIndex([t0])

        # window forecasts
        Xw = window_rows(feats, issues, lat, lon, clim)
        persist = h["pm25"].tail(24).mean()
        Pw = np.expm1(self.predict_window_log(Xw, np.full(3, persist)))

        # hourly curve, rescaled to the window forecasts
        horizons = np.arange(1, F.MAX_HORIZON + 1)
        Xh = F.build_rows(feats, pd.DatetimeIndex([t0] * len(horizons)), horizons, lat, lon, clim)
        Ph = np.expm1(self.predict_hourly_log(Xh))
        win_of_h = (horizons - 1) // 24
        factor_w = np.array([Pw[w, 1] / max(Ph[win_of_h == w, 1].mean(), 1e-6) for w in range(3)])
        centers = np.array([12.5, 36.5, 60.5])
        factor = np.interp(horizons, centers, factor_w)
        width_lo = np.array([Pw[w, 1] / max(Pw[w, 0], 1e-6) for w in range(3)])
        width_hi = np.array([Pw[w, 2] / max(Pw[w, 1], 1e-6) for w in range(3)])
        p50 = Ph[:, 1] * factor
        # hourly bands: at least as wide as the hourly model, and as wide as the calibrated window band
        lo_ratio = np.maximum(Ph[:, 1] / np.maximum(Ph[:, 0], 1e-6), np.interp(horizons, centers, width_lo))
        hi_ratio = np.maximum(Ph[:, 2] / np.maximum(Ph[:, 1], 1e-6), np.interp(horizons, centers, width_hi))
        hourly = pd.DataFrame({"time": t0 + pd.to_timedelta(horizons, unit="h"), "h": horizons,
                               "p10": p50 / lo_ratio, "p50": p50, "p90": p50 * hi_ratio})

        windows = []
        for w in range(3):
            p10, p50w, p90 = (float(v) for v in Pw[w])
            a = float(aqi_mod.subindex("pm25", p50w))
            sub = hourly[win_of_h == w]
            windows.append({
                "window": WINDOW_LABELS[w],
                "start": sub["time"].min(), "end": sub["time"].max(),
                "pm25_p10": p10, "pm25_p50": p50w, "pm25_p90": p90,
                "aqi": a, "aqi_p90": float(aqi_mod.subindex("pm25", p90)),
                "category": aqi_mod.category(a),
                "prob_poor": float(exceed_prob(p10, p50w, p90, 90)),
                "prob_very_poor": float(exceed_prob(p10, p50w, p90, 120)),
                "prob_severe": float(exceed_prob(p10, p50w, p90, 250)),
                "peak_time": sub.loc[sub["p50"].idxmax(), "time"],
                "peak_hour_pm25": float(sub["p50"].max()),
            })
        return {"hourly": hourly, "windows": windows, "persistence_24h": float(persist),
                "issue_time": t0, "city": city}

    # ------------------------------------------------------------------- io
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "Forecaster":
        return joblib.load(path)


def hybrid_with_physics(ml: dict, physics_hourly: pd.DataFrame | None, w_ml: float = 0.5) -> dict:
    """Blend the ML forecast with a physics-based forecast (e.g. Copernicus CAMS), geometric mean.

    physics_hourly: DataFrame with columns time, pm25 (hourly).
    """
    if physics_hourly is None or physics_hourly.empty:
        return ml
    hourly = ml["hourly"].copy()
    ph = physics_hourly.set_index("time")["pm25"].reindex(hourly["time"]).to_numpy(dtype=float)
    ok = ~np.isnan(ph)
    for col in ("p10", "p50", "p90"):
        v = hourly[col].to_numpy(dtype=float)
        ratio = np.where(ok, np.exp((1 - w_ml) * (np.log1p(ph) - np.log1p(hourly["p50"].to_numpy()))), 1.0)
        hourly[col] = np.where(ok, (1 + v) * ratio - 1, v)
    hourly["physics_pm25"] = ph
    out = dict(ml)
    out["hourly"] = hourly
    wins = []
    for w, win in enumerate(ml["windows"]):
        sub = hourly[(hourly["h"] > 24 * w) & (hourly["h"] <= 24 * (w + 1))]
        p10, p50, p90 = sub["p10"].mean(), sub["p50"].mean(), sub["p90"].mean()
        a = float(aqi_mod.subindex("pm25", p50))
        nw = dict(win)
        nw.update({"pm25_p10": p10, "pm25_p50": p50, "pm25_p90": p90, "aqi": a,
                   "aqi_p90": float(aqi_mod.subindex("pm25", p90)), "category": aqi_mod.category(a),
                   "prob_poor": float(exceed_prob(p10, p50, p90, 90)),
                   "prob_very_poor": float(exceed_prob(p10, p50, p90, 120)),
                   "prob_severe": float(exceed_prob(p10, p50, p90, 250))})
        wins.append(nw)
    out["windows"] = wins
    out["hybrid"] = True
    return out
