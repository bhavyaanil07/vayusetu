"""
Federated-learning experiment on real CPCB data: can Indian cities build a
better forecasting model together without sharing raw data?

Compares on each city's out-of-sample year (Jul 2019 - Jun 2020):
  * persistence          - tomorrow = last 24 h
  * local-only           - each city trains alone on its own data
  * federated (FedAvg)   - cities train together, only weights are shared
  * centralised          - all raw data pooled (privacy-violating upper bound)
plus cold-start (a city joining with only 14 days of data), zero-shot cities
never seen in training, and differentially-private FedAvg.

Outputs
  data/models/federated_global.npz, data/models/federated_card.json,
  data/models/federated_results.json, data/models/fl_demo_clients.npz
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

from vayusetu import config  # noqa: E402
from vayusetu.data import cpcb  # noqa: E402
from vayusetu.models import federated as FL  # noqa: E402
from vayusetu.models import features as F  # noqa: E402

TRAIN_END = pd.Timestamp("2019-05-30 23:00")
TEST_START, TEST_END = pd.Timestamp("2019-07-01"), pd.Timestamp("2020-06-29 23:00")
ROUNDS, LOCAL_EPOCHS, HIDDEN = 40, 2, 32
COLD_START_CITIES = ["Lucknow", "Kolkata", "Bengaluru"]


def city_dataset(city: str):
    meta = config.CITIES[city]
    df = cpcb.city_frame(city)
    feats = F.issue_features(df)
    lp = np.log1p(df["pm25"])
    ok = (lp.notna() & (feats["valid24"] >= 12)).values
    target = np.log1p(df["pm25"].shift(-24).rolling(24, min_periods=18).mean())
    out = {}
    for split, sel in (("train", (df.index <= TRAIN_END) & (df.index.hour % 6 == 0)),
                       ("test", (df.index >= TEST_START) & (df.index <= TEST_END) & (df.index.hour == 18))):
        t = df.index[sel & ok]
        if len(t) == 0:
            out[split] = None
            continue
        X = F.build_rows(feats, t, np.full(len(t), 12), meta.lat, meta.lon, None)
        base = X["lp_mean24"].fillna(X["lp_0"]).values
        y = target.reindex(t).values - base
        keep = ~np.isnan(y)
        out[split] = (X[FL.FL_FEATURES].values[keep].astype(float), y[keep].astype(np.float32),
                      base[keep], t[keep],
                      df["pm25"].rolling(24, min_periods=12).mean().reindex(t[keep]).values)
    return out


def mae_ug(w, Xp, y, base):
    pred = np.expm1(FL.predict(w, Xp) + base)
    obs = np.expm1(y + base)
    return float(np.mean(np.abs(pred - obs)))


def main():
    t0 = time.time()
    cities = cpcb.cities_with_data()
    data = {c: city_dataset(c) for c in cities}
    train_clients = [c for c in cities if data[c]["train"] is not None and len(data[c]["train"][1]) >= 300]
    print(f"Train clients ({len(train_clients)}): {train_clients}  ({time.time() - t0:.0f}s)")
    flag_idx = [FL.FL_FEATURES.index(f) for f in FL.MISSING_FLAGS]

    def fed_stats(names):
        return FL.combine_stats([FL.local_stats(data[c]["train"][0]) for c in names])

    mean, std = fed_stats(train_clients)
    prep = lambda X: FL.prepare(X, mean, std, flag_idx)  # noqa: E731
    clients = {c: (prep(data[c]["train"][0]), data[c]["train"][1]) for c in train_clients}
    tests = {c: (prep(data[c]["test"][0]), data[c]["test"][1], data[c]["test"][2], data[c]["test"][4])
             for c in cities if data[c]["test"] is not None and len(data[c]["test"][1]) >= 30}
    eval_sets = {c: (v[0], v[1]) for c, v in tests.items()}

    # ---------------- FedAvg
    w_fed, hist = FL.run_fedavg(clients, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, d_hidden=HIDDEN,
                                eval_sets=eval_sets, seed=1)
    print(f"FedAvg done ({time.time() - t0:.0f}s); final eval loss {hist[-1]['eval_loss']:.4f}")

    # ---------------- centralised
    Xall = np.vstack([v[0] for v in clients.values()])
    yall = np.concatenate([v[1] for v in clients.values()])
    w_cen = FL.train_local_only(Xall, yall, epochs=30, d_hidden=HIDDEN, seed=1)

    # ---------------- local-only
    local_models = {c: FL.train_local_only(*clients[c], epochs=30, d_hidden=HIDDEN, seed=1) for c in train_clients}
    print(f"Baselines trained ({time.time() - t0:.0f}s)")

    per_city = []
    for c, (Xp, y, base, per24) in tests.items():
        obs = np.expm1(y + base)
        row = {"city": c, "n_test": int(len(y)), "in_federation": c in train_clients,
               "train_samples": int(len(clients[c][1])) if c in clients else 0,
               "mae_persistence": float(np.nanmean(np.abs(per24 - obs))),
               "mae_federated": mae_ug(w_fed, Xp, y, base),
               "mae_centralised": mae_ug(w_cen, Xp, y, base),
               "mae_local_only": mae_ug(local_models[c], Xp, y, base) if c in local_models else None}
        per_city.append(row)
    pc = pd.DataFrame(per_city)
    print(pc.round(2).to_string())
    seen = pc[pc["in_federation"]]
    unseen = pc[~pc["in_federation"]]
    summary = {
        "federated_cities": len(train_clients),
        "mean_mae_persistence_seen": float(seen["mae_persistence"].mean()),
        "mean_mae_local_only_seen": float(seen["mae_local_only"].mean()),
        "mean_mae_federated_seen": float(seen["mae_federated"].mean()),
        "mean_mae_centralised_seen": float(seen["mae_centralised"].mean()),
        "cities_where_federated_beats_local": int((seen["mae_federated"] < seen["mae_local_only"]).sum()),
        "mean_mae_persistence_unseen": float(unseen["mae_persistence"].mean()) if len(unseen) else None,
        "mean_mae_federated_unseen": float(unseen["mae_federated"].mean()) if len(unseen) else None,
        "unseen_cities": unseen["city"].tolist(),
    }
    print(json.dumps(summary, indent=1))

    # ---------------- cold start: a city joins with only 14 days of data
    cold = []
    for c in COLD_START_CITIES:
        others = [x for x in train_clients if x != c]
        m2, s2 = fed_stats(others)
        prep2 = lambda X: FL.prepare(X, m2, s2, flag_idx)  # noqa: E731
        cl2 = {x: (prep2(data[x]["train"][0]), data[x]["train"][1]) for x in others}
        w_g, _ = FL.run_fedavg(cl2, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, d_hidden=HIDDEN, seed=2)
        Xtr, ytr, _, ttr, _ = data[c]["train"]
        sel = (ttr >= pd.Timestamp("2018-11-01")) & (ttr < pd.Timestamp("2018-11-15"))
        Xs, ys = prep2(Xtr[sel]), ytr[sel]
        w_loc = FL.train_local_only(Xs, ys, epochs=60, d_hidden=HIDDEN, seed=3)
        w_ft = FL.train_local(w_g, Xs, ys, epochs=3, lr=5e-4, seed=3)
        Xt, yt, bt, pt = data[c]["test"][0], data[c]["test"][1], data[c]["test"][2], data[c]["test"][4]
        Xtp = prep2(Xt)
        obs = np.expm1(yt + bt)
        cold.append({"city": c, "days_of_local_data": 14, "samples": int(sel.sum()),
                     "mae_persistence": float(np.nanmean(np.abs(pt - obs))),
                     "mae_local_14d": mae_ug(w_loc, Xtp, yt, bt),
                     "mae_federated_zero_shot": mae_ug(w_g, Xtp, yt, bt),
                     "mae_federated_finetuned": mae_ug(w_ft, Xtp, yt, bt)})
        print("cold start", cold[-1])

    # ---------------- differential privacy trade-off
    dp = []
    Xp_all = np.vstack([v[0] for v in tests.values()])
    y_all = np.concatenate([v[1] for v in tests.values()])
    b_all = np.concatenate([v[2] for v in tests.values()])
    dp.append({"noise_multiplier": 0.0, "clip": None, "mae": mae_ug(w_fed, Xp_all, y_all, b_all)})
    for nm in (0.5, 1.0):
        w_dp, _ = FL.run_fedavg(clients, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, d_hidden=HIDDEN,
                                clip=1.0, noise_multiplier=nm, seed=1)
        dp.append({"noise_multiplier": nm, "clip": 1.0, "mae": mae_ug(w_dp, Xp_all, y_all, b_all)})
        print("DP", dp[-1])

    # ---------------- save
    np.savez_compressed(config.MODELS_DIR / "federated_global.npz", **FL.weights_to_npz_dict(w_fed, mean, std),
                        features=np.array(FL.FL_FEATURES), flags=np.array(FL.MISSING_FLAGS))
    demo_cities = [c for c in ["Delhi", "Gurugram", "Lucknow", "Patna", "Kolkata", "Jaipur", "Ahmedabad", "Mumbai",
                               "Hyderabad", "Bengaluru", "Chennai", "Thiruvananthapuram", "Visakhapatnam", "Amritsar"]
                   if c in clients]
    rng = np.random.default_rng(0)
    demo = {}
    for c in demo_cities:
        X, y = clients[c]
        idx = rng.choice(len(y), size=min(700, len(y)), replace=False)
        demo[f"{c}__X"] = X[idx].astype(np.float16)
        demo[f"{c}__y"] = y[idx].astype(np.float16)
    ev_idx = rng.choice(len(y_all), size=min(3000, len(y_all)), replace=False)
    demo["__eval_X"] = Xp_all[ev_idx].astype(np.float16)
    demo["__eval_y"] = y_all[ev_idx].astype(np.float16)
    demo["__eval_b"] = b_all[ev_idx].astype(np.float32)
    np.savez_compressed(config.MODELS_DIR / "fl_demo_clients.npz", **demo)

    results = {"history": hist, "per_city": per_city, "summary": summary, "cold_start": cold, "dp": dp,
               "config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "hidden": HIDDEN,
                          "features": FL.FL_FEATURES, "train_clients": train_clients}}
    (config.MODELS_DIR / "federated_results.json").write_text(json.dumps(results, indent=2, default=str))
    card = {
        "model_id": "vayusetu-fl-global-v1", "name": "VayuSetu federated next-day PM2.5 model", "version": "1.0.0",
        "algorithm": f"FedAvg, MLP ({len(FL.FL_FEATURES) + len(FL.MISSING_FLAGS)}-{HIDDEN}-1), Huber loss, "
                     f"{ROUNDS} rounds x {LOCAL_EPOCHS} local epochs",
        "participants": train_clients, "data_shared": "Model weights and feature sums only - no raw measurements",
        "training_data_period": ["2015-01-01", str(TRAIN_END.date())],
        "evaluation": f"Out-of-sample {TEST_START.date()} to {TEST_END.date()} (next-24 h mean PM2.5)",
        "headline_metrics": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()},
        "license": "MIT", "format": "NumPy .npz (W1,b1,W2,b2,mean,std,features)",
    }
    (config.MODELS_DIR / "federated_card.json").write_text(json.dumps(card, indent=2))
    print(f"Saved. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
