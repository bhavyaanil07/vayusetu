"""
Train and evaluate VayuCast on real CPCB hourly data (26 cities).

Periods
  fit A        2015-01 .. 2018-11   proxy model used for calibration / blend tuning
  calibration  2018-12 .. 2019-05   conformal interval widths + blend weight
  fit final    2015-01 .. 2019-05   deployed model
  test         2019-07 .. 2020-06   out-of-sample evaluation (daily 18:00 IST issues)

Outputs
  data/models/forecaster.joblib, data/models/forecaster_card.json
  reports/forecast_metrics.json, reports/delhi_nov2019_backtest.csv
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

from vayusetu import aqi as A  # noqa: E402
from vayusetu import config  # noqa: E402
from vayusetu.data import cpcb  # noqa: E402
from vayusetu.models import features as F  # noqa: E402
from vayusetu.models import forecaster as FC  # noqa: E402

FIT_A_END = pd.Timestamp("2018-11-30 23:00")
CAL_START, CAL_END = pd.Timestamp("2018-12-01"), pd.Timestamp("2019-05-31 23:00")
TEST_START, TEST_END = pd.Timestamp("2019-07-01"), pd.Timestamp("2020-06-30 23:00")
COVID = (pd.Timestamp("2020-03-20"), pd.Timestamp("2020-05-31"))
RNG = np.random.default_rng(7)


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")


def build(frames, climatology, panel_all):
    """Build window + hourly datasets for all cities."""
    W, H = [], []
    for city, df in frames.items():
        meta = config.CITIES[city]
        clim = climatology.get(city)
        feats = F.issue_features(df)
        nb = FC.neighbour_features(city, panel_all).reindex(df.index)
        feats = pd.concat([feats, nb], axis=1)
        lp = np.log1p(df["pm25"])
        ok = (lp.notna() & (feats["valid24"] >= 12)).values
        persist24 = df["pm25"].rolling(24, min_periods=12).mean()
        targets = {w: df["pm25"].shift(-(24 * (w + 1))).rolling(24, min_periods=18).mean() for w in range(3)}

        # ---- window rows: training issues every 3 h, test/cal issues at 18:00
        for split, sel in (("train", (df.index.hour % 3 == 0)), ("eval", (df.index.hour == 18))):
            t = df.index[sel & ok]
            if len(t) == 0:
                continue
            X = FC.window_rows(feats, t, meta.lat, meta.lon, clim)
            n = len(t)
            X["_y"] = np.concatenate([np.log1p(targets[w].reindex(t).values) for w in range(3)])
            X["_issue"] = np.tile(t.values, 3)
            X["_city"] = city
            X["_persist24"] = np.tile(persist24.reindex(t).values, 3)
            X["_split"] = split
            W.append(X[~np.isnan(X["_y"].values)])

        # ---- hourly rows: every valid hour, two random horizons
        t = df.index[ok & (df.index <= CAL_END - pd.Timedelta(hours=72))]
        for _ in range(2):
            hz = RNG.integers(1, F.MAX_HORIZON + 1, len(t))
            X = F.build_rows(feats, t, hz, meta.lat, meta.lon, clim)
            X["_y"] = lp.reindex(t + pd.to_timedelta(hz, unit="h")).values
            X["_issue"] = t.values
            H.append(X[~np.isnan(X["_y"].values)])
        # ---- hourly test rows (daily 18:00 issues, all horizons) for hourly metrics
        tt = df.index[ok & (df.index.hour == 18) & (df.index >= TEST_START) & (df.index <= TEST_END - pd.Timedelta(hours=72))]
        if len(tt):
            hz = np.tile(np.arange(1, F.MAX_HORIZON + 1), len(tt))
            iss = pd.DatetimeIndex(tt.repeat(F.MAX_HORIZON))
            X = F.build_rows(feats, iss, hz, meta.lat, meta.lon, clim)
            X["_y"] = np.log1p(df["pm25"].reindex(iss + pd.to_timedelta(hz, unit="h")).values)
            X["_issue"] = iss.values
            X["_persist"] = df["pm25"].reindex(iss).values
            X["_city"] = city
            X["_test"] = True
            H.append(X)
    Wdf = pd.concat(W, ignore_index=True)
    Hdf = pd.concat(H, ignore_index=True)
    return Wdf, Hdf


def window_metrics(P, obs, persist, wins, issues):
    """P: (n,3) ug/m3 quantiles; returns metrics per window, all and excl. COVID lockdown."""
    res = {}
    covid = (issues >= COVID[0]) & (issues <= COVID[1])
    for tag, mask0 in (("all", np.ones(len(obs), bool)), ("excl_covid_lockdown", ~covid)):
        out = {}
        for w in range(3):
            m = mask0 & (wins == w) & ~np.isnan(persist)
            p10, p50, p90, o, pr = P[m, 0], P[m, 1], P[m, 2], obs[m], persist[m]
            oc = np.array([A.category_index(A.subindex("pm25", v)) for v in o])
            mc = np.array([A.category_index(A.subindex("pm25", v)) for v in p50])
            pc = np.array([A.category_index(A.subindex("pm25", v)) for v in pr])
            prob = FC.exceed_prob(p10, p50, p90, 120)
            ev = o > 120
            pred, ppred = prob >= 0.5, pr > 120

            def prf(pd_, ev_):
                tp, fp, fn = int((pd_ & ev_).sum()), int((pd_ & ~ev_).sum()), int((~pd_ & ev_).sum())
                p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                return p, r, 2 * p * r / max(p + r, 1e-9)

            mp, mr, mf = prf(pred, ev)
            pp, prr, pf = prf(ppred, ev)
            out[w] = {"n": int(m.sum()), "mae_model": mae(p50, o), "mae_persistence": mae(pr, o),
                      "skill_vs_persistence": 1 - mae(p50, o) / mae(pr, o),
                      "category_accuracy_model": float((oc == mc).mean()),
                      "category_accuracy_persistence": float((oc == pc).mean()),
                      "within_one_category_model": float((np.abs(oc - mc) <= 1).mean()),
                      "very_poor_events": int(ev.sum()),
                      "very_poor_precision": mp, "very_poor_recall": mr, "very_poor_f1": mf,
                      "very_poor_precision_persistence": pp, "very_poor_recall_persistence": prr,
                      "very_poor_f1_persistence": pf,
                      "brier_very_poor": float(np.mean((prob - ev) ** 2)),
                      "p10_p90_coverage": float(((o >= p10) & (o <= p90)).mean())}
        res[tag] = out
    return res


def main() -> None:
    t0 = time.time()
    cities = cpcb.cities_with_data()
    frames = {c: cpcb.city_frame(c) for c in cities}
    climatology = {}
    for c in cities:
        tp = frames[c].loc[:FIT_A_END, "pm25"]
        if tp.notna().sum() > 24 * 90:
            climatology[c] = F.climatology_table(tp)
    idx = pd.date_range("2015-01-01", "2020-07-01", freq="h")
    panel = FC.anomaly_panel(frames, climatology, idx)
    Wdf, Hdf = build(frames, climatology, panel)
    print(f"Datasets built in {time.time() - t0:.0f}s: window rows {len(Wdf):,}, hourly rows {len(Hdf):,}")

    wcols = [c for c in Wdf.columns if not c.startswith("_")]
    hcols = [c for c in Hdf.columns if not c.startswith("_")]
    iss = pd.to_datetime(Wdf["_issue"])
    tr_A = (Wdf["_split"] == "train") & (iss <= FIT_A_END - pd.Timedelta(hours=72))
    cal = (Wdf["_split"] == "eval") & (iss >= CAL_START) & (iss <= CAL_END - pd.Timedelta(hours=72))
    tr_F = (Wdf["_split"] == "train") & (iss <= CAL_END - pd.Timedelta(hours=72))
    test = (Wdf["_split"] == "eval") & (iss >= TEST_START) & (iss <= TEST_END - pd.Timedelta(hours=72))

    # ---------------- proxy model A -> blend weight + conformal widths
    mA = FC.Forecaster(climatology=climatology)
    mA.fit_window(Wdf.loc[tr_A, wcols], Wdf.loc[tr_A, "_y"].values)
    Xc = Wdf.loc[cal, wcols]
    yc = Wdf.loc[cal, "_y"].values
    pc = Wdf.loc[cal, "_persist24"].values
    best = None
    for w in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        mA.blend_weight = w
        P = mA.predict_window_log(Xc, pc, calibrated=False)
        err = mae(np.expm1(P[:, 1]), np.expm1(yc))
        print(f"  blend weight {w:.1f}: calibration MAE {err:.2f}")
        if best is None or err < best[1]:
            best = (w, err)
    mA.blend_weight = best[0]
    P = mA.predict_window_log(Xc, pc, calibrated=False)
    conformal = {}
    wins_c = Xc["win"].values.astype(int)
    for w in range(3):
        m = wins_c == w
        lo_scores = P[m, 0] - yc[m]          # >0 when obs falls below p10
        hi_scores = yc[m] - P[m, 2]          # >0 when obs falls above p90
        n = m.sum()
        level = min(1.0, np.ceil((n + 1) * 0.9) / n)
        conformal[w] = (float(max(np.quantile(lo_scores, level), 0.0)), float(max(np.quantile(hi_scores, level), 0.0)))
    print("  blend weight:", best[0], "conformal widening (log):", {k: tuple(round(x, 3) for x in v) for k, v in conformal.items()})

    # ---------------- final model
    model = FC.Forecaster(climatology=climatology, blend_weight=best[0], conformal=conformal)
    model.fit_window(Wdf.loc[tr_F, wcols], Wdf.loc[tr_F, "_y"].values)
    htr = ~Hdf.get("_test", pd.Series(False, index=Hdf.index)).fillna(False).astype(bool)
    model.fit_hourly(Hdf.loc[htr, hcols], Hdf.loc[htr, "_y"].values)
    print(f"Final models trained ({time.time() - t0:.0f}s)")

    # ---------------- window evaluation
    Xt = Wdf.loc[test, wcols]
    obs = np.expm1(Wdf.loc[test, "_y"].values)
    per = Wdf.loc[test, "_persist24"].values
    wins = Xt["win"].values.astype(int)
    issues_t = pd.to_datetime(Wdf.loc[test, "_issue"]).values
    P_final = np.expm1(model.predict_window_log(Xt, per))
    P_raw = np.expm1(model.predict_window_log(Xt, None, calibrated=False))
    wm = window_metrics(P_final, obs, per, wins, issues_t)
    wm_raw = window_metrics(P_raw, obs, per, wins, issues_t)
    for tag in ("all", "excl_covid_lockdown"):
        for w in range(3):
            r = wm[tag][w]
            print(f"[{tag}] win {w}: MAE {r['mae_model']:.1f} vs persist {r['mae_persistence']:.1f} "
                  f"(skill {r['skill_vs_persistence']:+.3f}) | cat {r['category_accuracy_model']:.2f} vs "
                  f"{r['category_accuracy_persistence']:.2f} | VP+ F1 {r['very_poor_f1']:.2f} vs {r['very_poor_f1_persistence']:.2f} "
                  f"| Brier {r['brier_very_poor']:.3f} | cover {r['p10_p90_coverage']:.2f}")

    # reliability (window 0)
    m0 = (wins == 0) & ~np.isnan(per)
    prob0 = FC.exceed_prob(P_final[m0, 0], P_final[m0, 1], P_final[m0, 2], 120)
    ev0 = obs[m0] > 120
    bins = np.linspace(0, 1, 11)
    rel = []
    for i in range(10):
        s = (prob0 >= bins[i]) & (prob0 < bins[i + 1] if i < 9 else prob0 <= 1)
        if s.sum() >= 10:
            rel.append({"bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "mean_prob": float(prob0[s].mean()),
                        "observed_freq": float(ev0[s].mean()), "n": int(s.sum())})

    # per-city next-24h
    dfc = pd.DataFrame({"city": Wdf.loc[test, "_city"].values, "win": wins, "obs": obs, "p50": P_final[:, 1], "per": per})
    dfc = dfc[(dfc["win"] == 0) & dfc["per"].notna()]
    per_city = dfc.groupby("city").apply(lambda s: pd.Series({
        "mae_model": mae(s["p50"], s["obs"]), "mae_persistence": mae(s["per"], s["obs"]), "n": len(s)}),
        include_groups=False).reset_index()
    per_city["skill"] = 1 - per_city["mae_model"] / per_city["mae_persistence"]
    per_city["seen_in_training"] = per_city["city"].isin(list(climatology.keys()))
    print(per_city.round(3).to_string())

    # ---------------- hourly evaluation (raw hourly model)
    Ht = Hdf[Hdf.get("_test", False).fillna(False).astype(bool)]
    Ph = np.expm1(model.predict_hourly_log(Ht[hcols]))
    yh = np.expm1(Ht["_y"].values)
    hourly = {}
    for h in [1, 3, 6, 12, 24, 48, 72]:
        m = Ht["h"].values == h
        hourly[h] = {"mae_model": mae(Ph[m, 1], yh[m]), "mae_persistence": mae(Ht["_persist"].values[m], yh[m])}
        hourly[h]["skill_vs_persistence"] = 1 - hourly[h]["mae_model"] / hourly[h]["mae_persistence"]
        print(f"hourly h={h:>2}: MAE {hourly[h]['mae_model']:.1f} vs persistence {hourly[h]['mae_persistence']:.1f}")

    # ---------------- permutation importance (window median model)
    from sklearn.inspection import permutation_importance
    samp = Xt.sample(min(15000, len(Xt)), random_state=1)
    ys = np.log1p(obs[Xt.index.get_indexer(samp.index)]) - FC._base(samp)
    pi = permutation_importance(model.window_models[0.5], samp.values.astype(np.float32), ys,
                                n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
    importance = sorted(zip(wcols, pi.importances_mean), key=lambda x: -x[1])

    # ---------------- Delhi Oct-Nov 2019 backtest through the full inference path
    rows = []
    for day in pd.date_range("2019-10-22 18:00", "2019-11-16 18:00", freq="D"):
        fc = model.forecast(frames, "Delhi", day)
        for w in fc["windows"]:
            actual = frames["Delhi"]["pm25"].loc[w["start"]:w["end"]].mean()
            rows.append({"issue": day, "window": w["window"], "start": w["start"], "pred_p10": w["pm25_p10"],
                         "pred_p50": w["pm25_p50"], "pred_p90": w["pm25_p90"], "prob_very_poor": w["prob_very_poor"],
                         "prob_severe": w["prob_severe"], "actual": actual, "persistence": fc["persistence_24h"]})
    bt = pd.DataFrame(rows)
    bt.to_csv(config.REPORTS_DIR / "delhi_nov2019_backtest.csv", index=False)
    print(bt[bt["window"] == "24-48 h"][["issue", "pred_p50", "prob_very_poor", "prob_severe", "actual", "persistence"]].round(2).to_string())

    model.meta = {"name": "VayuCast", "version": "1.1.0",
                  "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                  "train_period": ["2015-01-01", str(CAL_END.date())],
                  "test_period": [str(TEST_START.date()), str(TEST_END.date())],
                  "cities_with_climatology": sorted(climatology.keys()),
                  "window_features": wcols, "hourly_features": hcols}
    path = config.MODELS_DIR / "forecaster.joblib"
    model.save(path)
    size_mb = path.stat().st_size / 1e6

    metrics = {"windows": wm, "windows_uncalibrated": wm_raw, "hourly": hourly, "reliability_next24h": rel,
               "per_city_next24h": per_city.to_dict(orient="records"),
               "importance_window_model": [{"feature": f, "importance": float(v)} for f, v in importance],
               "blend_weight": best[0], "conformal": conformal, "model_size_mb": size_mb}
    (config.REPORTS_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    w0 = wm["excl_covid_lockdown"]
    card = {
        "model_id": "vayucast-pm25-v1.1", "name": "VayuCast PM2.5 probabilistic forecaster", "version": "1.1.0",
        "task": "PM2.5 forecast 1-72 h ahead: hourly curve + 24-h average windows with 10/50/90 % quantiles",
        "algorithm": "Histogram gradient boosting (scikit-learn), quantile loss; blended with persistence; conformal calibration",
        "training_data": "CPCB hourly city-level data, 26 Indian cities, 2015-01 to 2019-05",
        "evaluation": "Out-of-sample Jul 2019 - Jun 2020, one issue per day at 18:00 IST",
        "headline_metrics": {
            "skill_vs_persistence_next24h": round(w0[0]["skill_vs_persistence"], 3),
            "skill_vs_persistence_24_48h": round(w0[1]["skill_vs_persistence"], 3),
            "skill_vs_persistence_48_72h": round(w0[2]["skill_vs_persistence"], 3),
            "mae_next24h_ugm3": round(w0[0]["mae_model"], 1),
            "very_poor_f1_next24h": round(w0[0]["very_poor_f1"], 3),
            "p10_p90_coverage_next24h": round(w0[0]["p10_p90_coverage"], 3),
            "note": "metrics exclude the COVID-19 lockdown (20 Mar - 31 May 2020); full-period metrics in reports/forecast_metrics.json",
        },
        "inputs": "Recent PM2.5/PM10/NO2/CO history, climatology prior, calendar & Diwali effects, neighbouring-city pollution",
        "intended_use": "Early warning of PM2.5 spikes for city/state authorities (decision support, not a regulatory measurement)",
        "limitations": ["Trained on city-average station data; ward-level detail comes from the citizen-sensor layer",
                        "No direct meteorological inputs in v1; live mode blends with Copernicus CAMS physics forecasts",
                        "Performance is lower for cities first seen after training (no climatology prior)"],
        "owner_node": "VayuSetu national hub", "license": "MIT", "size_mb": round(size_mb, 2),
    }
    (config.MODELS_DIR / "forecaster_card.json").write_text(json.dumps(card, indent=2))
    print(f"Saved model ({size_mb:.1f} MB). Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
