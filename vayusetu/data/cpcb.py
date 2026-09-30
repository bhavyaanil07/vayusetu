"""Access to the bundled CPCB hourly dataset (26 cities, 2015-2020, IST timestamps)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..config import CPCB_PARQUET

POLLUTANT_COLS = ["pm25", "pm10", "no2", "co", "so2", "o3"]


@lru_cache(maxsize=1)
def load_cpcb() -> pd.DataFrame:
    df = pd.read_parquet(CPCB_PARQUET)
    df["city"] = df["city"].astype(str)
    return df


@lru_cache(maxsize=64)
def city_frame(city: str) -> pd.DataFrame:
    df = load_cpcb()
    g = df[df["city"] == city].set_index("time").sort_index()[POLLUTANT_COLS]
    if g.empty:
        raise KeyError(f"No CPCB data for {city}")
    full = pd.date_range(g.index.min(), g.index.max(), freq="h")
    return g.reindex(full).astype(float)


def cities_with_data() -> list[str]:
    return sorted(load_cpcb()["city"].unique().tolist())


def snapshot(time, window_hours: int = 24) -> pd.DataFrame:
    """Latest hourly value and 24-h means for every city at `time` (replay mode)."""
    time = pd.Timestamp(time)
    rows = []
    for city in cities_with_data():
        g = city_frame(city)
        w = g.loc[time - pd.Timedelta(hours=window_hours - 1): time]
        if w.empty:
            continue
        last = w.ffill().iloc[-1]
        rows.append({
            "city": city, "time": time,
            "pm25": last["pm25"], "pm10": last["pm10"], "no2": last["no2"], "co": last["co"],
            "pm25_24h": w["pm25"].mean(), "pm10_24h": w["pm10"].mean(),
            "valid_hours": int(w["pm25"].notna().sum()),
        })
    return pd.DataFrame(rows)
