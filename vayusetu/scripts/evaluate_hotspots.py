"""
Evaluate hidden-hotspot detection on the Delhi Oct-Nov 2019 replay.

Background pollution is real (CPCB); local events and sensor readings are
simulated (see vayusetu/data/citizen.py), so detections can be scored
against known ground truth.

Compares three monitoring designs using the same algorithm:
  * VayuSetu citizen network  - ~1 low-cost sensor per ward (+ priority sites)
  * Regulatory-density network - 40 sites (similar to Delhi's CAAQMS count)
  * City-average monitoring    - one number per city (cannot localise anything)

Output: reports/hotspot_eval.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.cluster import KMeans  # noqa: E402

from vayusetu import config  # noqa: E402
from vayusetu.data import citizen, cpcb  # noqa: E402
from vayusetu.data.citizen import km_offsets  # noqa: E402
from vayusetu.models import hotspots as H  # noqa: E402

CITY = "Delhi"
START, END = pd.Timestamp(config.REPLAY_START), pd.Timestamp(config.REPLAY_END)
STEP_H = 2
MATCH_KM = 2.5


def _plume_match(e, h, wind_from: float) -> float:
    """Distance score: 0 if the hotspot lies inside the event's plume footprint, else km."""
    dx, dy = km_offsets(e.lat, e.lon, h["lat"], h["lon"])
    d = float(np.hypot(dx, dy))
    to = np.radians((wind_from + 180) % 360)
    along = dx * np.sin(to) + dy * np.cos(to)
    cross = abs(-dx * np.cos(to) + dy * np.sin(to))
    if 0 <= along <= 6.0 and cross <= 2.5:
        return 0.3 * min(d, MATCH_KM)  # inside the plume footprint: always a match, closer still wins
    return d


def run(readings, sites, reports, events, times, met):
    rows = []
    for t in times:
        hs, _ = H.detect_hotspots(CITY, readings, sites, t, reports=reports)
        window = pd.date_range(t - pd.Timedelta(hours=2), t, freq="h")
        active = [e for e in events if e.active_factor(window).max() > 0]
        active_now = [e for e in events if e.active_factor(pd.DatetimeIndex([t]))[0] > 0]
        wind_from = float(met["wind_dir"].reindex([t]).iloc[0])
        detected_now, detected_conf = set(), set()
        for h in hs:
            thr = max(MATCH_KM, h["radius_km"])
            scores = [(_plume_match(e, h, wind_from), e) for e in active]
            hits = [e for d, e in scores if d <= thr]
            detected_now.update(e.id for e in hits)
            best = min(scores, key=lambda x: x[0])[1] if hits else None
            if h["confidence"] >= 0.6:
                detected_conf.update(e.id for e in hits)
            rows.append({"time": t, "type": "hotspot", "matched": bool(hits), "event": best.id if best else None,
                         "event_kind": best.kind if best else None, "pred_kind": h["source"],
                         "confidence": h["confidence"]})
        for e in active_now:
            rows.append({"time": t, "type": "active_event", "event": e.id, "event_kind": e.kind,
                         "detected": e.id in detected_now, "detected_conf": e.id in detected_conf, "start": e.start})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, events) -> dict:
    hs = df[df["type"] == "hotspot"]
    ae = df[df["type"] == "active_event"]
    ev_ids = ae["event"].unique()
    detected_ids = ae.loc[ae["detected"], "event"].unique()
    first = ae[ae["detected"]].groupby("event")["time"].min()
    starts = {e.id: e.start for e in events}
    delays = [(first[i] - max(starts[i], START)).total_seconds() / 3600 for i in first.index]
    matched = hs[hs["matched"]]
    by_kind = {}
    for k in config.SOURCE_TYPES:
        ids = ae.loc[ae["event_kind"] == k, "event"].unique()
        if len(ids):
            by_kind[k] = float(np.isin(ids, detected_ids).mean())
    return {
        "events_evaluated": int(len(ev_ids)),
        "event_recall": float(len(detected_ids) / max(len(ev_ids), 1)),
        "event_recall_by_kind": by_kind,
        "hourly_detection_rate": float(ae["detected"].mean()) if len(ae) else 0.0,
        "hotspots_raised": int(len(hs)),
        "precision": float(hs["matched"].mean()) if len(hs) else 0.0,
        "source_attribution_accuracy": float((matched["pred_kind"] == matched["event_kind"]).mean()) if len(matched) else 0.0,
        "median_detection_delay_h": float(np.median(delays)) if delays else None,
        "false_alarms_per_day": float((~hs["matched"].astype(bool)).sum() / ((END - START).total_seconds() / 86400)),
        "recall_at_alert_confidence": float(len(ae.loc[ae["detected_conf"].astype(bool), "event"].unique()) / max(len(ev_ids), 1)),
        "precision_at_alert_confidence": float(hs.loc[hs["confidence"] >= 0.6, "matched"].mean()) if (hs["confidence"] >= 0.6).any() else 0.0,
        "false_alarms_per_day_at_alert_confidence": float((~hs.loc[hs["confidence"] >= 0.6, "matched"].astype(bool)).sum() / ((END - START).total_seconds() / 86400)),
    }


def main() -> None:
    t0 = time.time()
    times_all = pd.date_range(START - pd.Timedelta(days=7), END, freq="h")
    bg = cpcb.city_frame(CITY)
    met = citizen.modeled_met(CITY, times_all, bg["pm25"].rolling(24, min_periods=6).mean())
    events = citizen.events_for(CITY, times_all[0], times_all[-1], replay=True)
    sites = citizen.load_sites(CITY)
    readings = citizen.simulate_readings(CITY, times_all, bg, met, events)
    reports = citizen.simulate_reports(CITY, events, times_all[0], times_all[-1])
    eval_times = pd.date_range(START, END, freq=f"{STEP_H}h")
    print(f"Simulated {len(readings):,} readings, {len(events)} events ({time.time() - t0:.0f}s)")

    full = run(readings, sites, reports, events, eval_times, met)
    full.to_csv(config.REPORTS_DIR / "hotspot_eval_detail.csv", index=False)
    res_full = summarise(full, events)
    print("citizen network:", json.dumps(res_full, indent=1))

    # regulatory-density network: 40 sites spread across the city (k-means over sensor sites)
    xy = np.column_stack(km_offsets(config.CITIES[CITY].lat, config.CITIES[CITY].lon, sites["lat"].values, sites["lon"].values))
    km = KMeans(n_clusters=40, n_init=5, random_state=0).fit(xy)
    chosen = sorted({int(np.argmin(((xy - c) ** 2).sum(1))) for c in km.cluster_centers_})
    sparse_sites = sites.iloc[chosen].reset_index(drop=True)
    sparse_readings = readings[readings["sensor_id"].isin(sparse_sites["id"])]
    sparse = run(sparse_readings, sparse_sites, None, events, eval_times, met)
    res_sparse = summarise(sparse, events)
    print("regulatory-density network:", json.dumps(res_sparse, indent=1))

    out = {
        "city": CITY, "period": [str(START), str(END)], "step_hours": STEP_H, "match_km": MATCH_KM,
        "background_data": "Real CPCB hourly PM2.5/PM10 (Delhi city average)",
        "simulated": "Local emission events, low-cost sensor readings (bias, humidity artefacts, noise, dropouts, 2 faulty units), photo reports",
        "citizen_network": {"sensors": int(len(sites)), **res_full},
        "regulatory_density_network": {"sensors": int(len(sparse_sites)), **res_sparse},
        "city_average_monitoring": {"sensors": 1, "event_recall": 0.0,
                                    "note": "A single city-wide value cannot localise any hotspot"},
    }
    (config.REPORTS_DIR / "hotspot_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
