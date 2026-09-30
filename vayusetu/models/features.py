"""
Feature engineering shared by the forecaster and the federated model.

Everything is computed from an hourly series of pollutant concentrations
(PM2.5, PM10, NO2, CO) so the same code works for CPCB station data,
citizen-sensor averages or Copernicus CAMS model output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DIWALI_DATES

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_HOURS = [3, 6, 12, 24, 72, 168]
MAX_HORIZON = 72

_DIWALI = pd.to_datetime(pd.Series(DIWALI_DATES)).values.astype("datetime64[D]")


def complete_hourly(df: pd.DataFrame, end=None) -> pd.DataFrame:
    """Reindex to a complete hourly index (missing hours become NaN)."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    last = pd.Timestamp(end).floor("h") if end is not None else df.index.max().floor("h")
    full = pd.date_range(df.index.min().floor("h"), last, freq="h")
    return df.reindex(full)


def issue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at issue time t. `df` must be hourly with pm25 (+ optional pm10/no2/co)."""
    lp = np.log1p(df["pm25"].astype(float))
    f = {"lp_0": lp}
    for lag in LAG_HOURS:
        f[f"lp_lag{lag}"] = lp.shift(lag)
    for w in ROLL_HOURS:
        f[f"lp_mean{w}"] = lp.rolling(w, min_periods=max(1, w // 3)).mean()
    f["lp_std24"] = lp.rolling(24, min_periods=8).std()
    f["lp_d24"] = lp - lp.shift(24)
    f["lp_trend6"] = lp.rolling(6, min_periods=3).mean() - lp.shift(6).rolling(6, min_periods=3).mean()
    for col, name in (("pm10", "lp10"), ("no2", "lno2"), ("co", "lco")):
        if col in df:
            s = np.log1p(df[col].astype(float))
            f[f"{name}_mean24"] = s.rolling(24, min_periods=8).mean()
        else:
            f[f"{name}_mean24"] = pd.Series(np.nan, index=df.index)
    if "pm10" in df:
        pm25_24 = df["pm25"].rolling(24, min_periods=8).mean()
        pm10_24 = df["pm10"].rolling(24, min_periods=8).mean()
        f["ratio_25_10"] = (pm25_24 / pm10_24).clip(0, 1.5)
    else:
        f["ratio_25_10"] = pd.Series(np.nan, index=df.index)
    f["valid24"] = lp.notna().rolling(24, min_periods=1).sum()
    f["hour_issue"] = pd.Series(df.index.hour, index=df.index)
    return pd.DataFrame(f, index=df.index)


def diwali_delta(times: pd.DatetimeIndex) -> np.ndarray:
    d = np.asarray(times.values.astype("datetime64[D]"))
    diff = (d[:, None] - _DIWALI[None, :]).astype(int)
    idx = np.abs(diff).argmin(axis=1)
    out = diff[np.arange(len(d)), idx].astype(float)
    return np.clip(out, -15, 15)


def target_features(target_times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    doy = target_times.dayofyear.values
    return {
        "hour_t": target_times.hour.values,
        "dow_t": target_times.dayofweek.values,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "month_t": target_times.month.values,
        "diwali_d": diwali_delta(target_times),
    }


def climatology_table(series: pd.Series) -> pd.DataFrame:
    """Mean log(PM2.5+1) by (month, hour) - a seasonal and diurnal prior."""
    lp = np.log1p(series.astype(float)).dropna()
    t = pd.DataFrame({"lp": lp.values, "month": lp.index.month, "hour": lp.index.hour})
    tab = t.groupby(["month", "hour"])["lp"].mean().unstack("hour")
    tab = tab.reindex(index=range(1, 13), columns=range(24))
    # Fill gaps (months without data) from neighbouring months.
    tab = tab.interpolate(axis=0, limit_direction="both")
    return tab


def clim_lookup(tab: pd.DataFrame | None, times: pd.DatetimeIndex) -> np.ndarray:
    if tab is None:
        return np.full(len(times), np.nan)
    arr = tab.values
    return arr[times.month.values - 1, times.hour.values]


def build_rows(feats: pd.DataFrame, issue_times: pd.DatetimeIndex, horizons: np.ndarray,
               lat: float, lon: float, clim: pd.DataFrame | None) -> pd.DataFrame:
    """Combine issue-time features with horizon/target-time features for (t, h) pairs."""
    X = feats.reindex(issue_times).reset_index(drop=True)
    target_times = issue_times + pd.to_timedelta(horizons, unit="h")
    X["h"] = horizons
    for k, v in target_features(pd.DatetimeIndex(target_times)).items():
        X[k] = v
    X["lat"] = lat
    X["lon"] = lon
    clim_t = clim_lookup(clim, pd.DatetimeIndex(target_times))
    clim_i = clim_lookup(clim, pd.DatetimeIndex(issue_times))
    X["clim_t"] = clim_t
    X["anom_0"] = X["lp_0"].values - clim_i
    X["anom_mean24"] = X["lp_mean24"].values - clim_i
    X["clim_shift"] = clim_t - clim_i
    return X
