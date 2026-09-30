"""
India's National Air Quality Index (NAQI, CPCB 2014).

The AQI of a place is the highest sub-index among the measured pollutants.
PM sub-indices use 24-hour averages, CO and O3 use 8-hour averages.
"""
from __future__ import annotations

import numpy as np

# (concentration breakpoints, index breakpoints). The last segment is
# extrapolated linearly and the final AQI is capped at 500.
_BREAKPOINTS = {
    "pm25": ([0, 30, 60, 90, 120, 250, 380], [0, 50, 100, 200, 300, 400, 500]),
    "pm10": ([0, 50, 100, 250, 350, 430, 510], [0, 50, 100, 200, 300, 400, 500]),
    "no2": ([0, 40, 80, 180, 280, 400, 520], [0, 50, 100, 200, 300, 400, 500]),
    "so2": ([0, 40, 80, 380, 800, 1600, 2400], [0, 50, 100, 200, 300, 400, 500]),
    "co": ([0, 1, 2, 10, 17, 34, 51], [0, 50, 100, 200, 300, 400, 500]),  # mg/m3
    "o3": ([0, 50, 100, 168, 208, 748, 1287], [0, 50, 100, 200, 300, 400, 500]),
}

CATEGORIES = [
    # name, lower AQI, upper AQI, hex colour, text colour
    ("Good", 0, 50, "#00A651", "#ffffff"),
    ("Satisfactory", 51, 100, "#8DC63F", "#1b1b1b"),
    ("Moderate", 101, 200, "#F2C200", "#1b1b1b"),
    ("Poor", 201, 300, "#F7941D", "#1b1b1b"),
    ("Very Poor", 301, 400, "#E53935", "#ffffff"),
    ("Severe", 401, 500, "#8B1A1A", "#ffffff"),
]

HEALTH_ADVICE = {
    "Good": "Minimal impact.",
    "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
    "Moderate": "May cause breathing discomfort to people with lung disease such as asthma, "
                "and discomfort to people with heart disease, children and older adults.",
    "Poor": "May cause breathing discomfort to people on prolonged exposure, and discomfort "
            "to people with heart disease.",
    "Very Poor": "May cause respiratory illness on prolonged exposure. Effects may be more "
                 "pronounced in people with lung and heart diseases.",
    "Severe": "May cause respiratory effects even in healthy people, and serious health "
              "impacts on people with lung or heart disease, even during light activity.",
}

# PM2.5 (24-h average, ug/m3) at which each category begins.
PM25_CATEGORY_START = {"Good": 0, "Satisfactory": 30.01, "Moderate": 60.01,
                       "Poor": 90.01, "Very Poor": 120.01, "Severe": 250.01}


def subindex(pollutant: str, conc):
    """Sub-index for one pollutant. Works on scalars and numpy arrays."""
    bp_c, bp_i = _BREAKPOINTS[pollutant]
    x = np.asarray(conc, dtype=float)
    out = np.interp(x, bp_c, bp_i)
    # Linear extrapolation above the last breakpoint.
    slope = (bp_i[-1] - bp_i[-2]) / (bp_c[-1] - bp_c[-2])
    over = x > bp_c[-1]
    out = np.where(over, bp_i[-1] + (x - bp_c[-1]) * slope, out)
    out = np.where(np.isnan(x), np.nan, out)
    out = np.clip(out, 0, 500)
    return out if out.ndim else float(out)


def aqi(pm25=None, pm10=None, no2=None, so2=None, co=None, o3=None):
    """Overall AQI = max of the available sub-indices (NaN-safe)."""
    parts = []
    for name, val in (("pm25", pm25), ("pm10", pm10), ("no2", no2),
                      ("so2", so2), ("co", co), ("o3", o3)):
        if val is not None:
            parts.append(np.atleast_1d(subindex(name, val)).astype(float))
    if not parts:
        raise ValueError("At least one pollutant is required")
    stacked = np.vstack(parts)
    all_nan = np.all(np.isnan(stacked), axis=0)
    filled = np.where(np.isnan(stacked), -1.0, stacked)
    res = np.where(all_nan, np.nan, filled.max(axis=0))
    return res if res.size > 1 else float(res[0])


def category(aqi_value) -> str:
    if aqi_value is None or (isinstance(aqi_value, float) and np.isnan(aqi_value)):
        return "No data"
    v = int(round(float(aqi_value)))
    for name, lo, hi, *_ in CATEGORIES:
        if v <= hi:
            return name
    return "Severe"


def category_index(aqi_value) -> int:
    names = [c[0] for c in CATEGORIES]
    cat = category(aqi_value)
    return names.index(cat) if cat in names else -1


def color_hex(aqi_value) -> str:
    cat = category(aqi_value)
    for name, _, _, hexc, _ in CATEGORIES:
        if name == cat:
            return hexc
    return "#9E9E9E"


def text_color(aqi_value) -> str:
    cat = category(aqi_value)
    for name, _, _, _, txt in CATEGORIES:
        if name == cat:
            return txt
    return "#ffffff"


def color_rgb(aqi_value, alpha: int = 200) -> list[int]:
    h = color_hex(aqi_value).lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def pm25_to_rgb(pm25_value, alpha: int = 200) -> list[int]:
    return color_rgb(subindex("pm25", pm25_value), alpha)


def health_advice(aqi_value) -> str:
    return HEALTH_ADVICE.get(category(aqi_value), "")


def grap_stage(aqi_value) -> int:
    """Graded Response Action Plan stage for Delhi-NCR (0 = not invoked)."""
    if aqi_value is None or np.isnan(aqi_value):
        return 0
    v = float(aqi_value)
    if v > 450:
        return 4
    if v > 400:
        return 3
    if v > 300:
        return 2
    if v > 200:
        return 1
    return 0


GRAP_ACTIONS = {
    1: ["Enforce dust-control norms at construction and demolition (C&D) sites",
        "Ban open burning of waste; step up mechanised road sweeping and water sprinkling",
        "Strict PUC checks; traffic management at congestion points",
        "Uninterrupted power supply to avoid diesel generator use"],
    2: ["Daily mechanised sweeping and dust suppression on major roads and hotspots",
        "Restrict diesel generator sets; augment public transport and raise parking fees",
        "Staggered timings for public offices; restrict large public events",
        "Targeted inspections of industries and hotspots flagged by VayuSetu"],
    3: ["Suspend non-essential construction and demolition",
        "Restrict older polluting vehicles (BS-III petrol / BS-IV diesel light vehicles)",
        "Hybrid classes for primary schools; 50% work-from-home advisories",
        "Close stone crushers, brick kilns and hot-mix plants in the region"],
    4: ["Stop entry of non-essential diesel trucks into the city",
        "Halt all C&D activity including linear public projects",
        "Move schools to online or hybrid mode; consider odd-even vehicle rationing",
        "Issue public health emergency advisory through the SACHET (CAP) alert system"],
}
