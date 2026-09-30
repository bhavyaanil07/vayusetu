"""
Federated learning across city nodes (FedAvg, McMahan et al. 2017).

Each city (or state) keeps its monitoring data on its own server. To build a
shared forecasting model they exchange only model weights:

  1. Federated normalisation - every node sends counts, sums and sums of
     squares per feature; the hub combines them into global means/stds.
  2. Each round the hub sends the current global model to the nodes; each
     node trains a few epochs on its own data and returns updated weights.
  3. The hub averages the weights (weighted by sample count).
  4. Optional privacy: clip each node's update and add Gaussian noise to the
     average (DP-FedAvg style), so no single city's data can be inferred.

The model is a small neural network (MLP) predicting the next-24-hour mean
PM2.5 (as a change from the last 24-hour mean, in log space).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FL_FEATURES = [
    "lp_0", "lp_lag1", "lp_lag3", "lp_lag6", "lp_lag12", "lp_lag24", "lp_lag48", "lp_lag72", "lp_lag168",
    "lp_mean3", "lp_mean6", "lp_mean12", "lp_mean24", "lp_mean72", "lp_mean168", "lp_std24", "lp_d24",
    "lp_trend6", "lp10_mean24", "lno2_mean24", "lco_mean24", "ratio_25_10", "hour_issue",
    "doy_sin", "doy_cos", "dow_t", "diwali_d",
]
MISSING_FLAGS = ["lp10_mean24", "lno2_mean24", "lco_mean24", "ratio_25_10"]


# ----------------------------------------------------------------------------
# Federated normalisation
# ----------------------------------------------------------------------------
def local_stats(X: np.ndarray) -> dict:
    m = ~np.isnan(X)
    Xz = np.where(m, X, 0.0)
    return {"n": m.sum(0).astype(float), "s": Xz.sum(0), "ss": (Xz ** 2).sum(0)}


def combine_stats(stats: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = sum(s["n"] for s in stats)
    s = sum(s["s"] for s in stats)
    ss = sum(s["ss"] for s in stats)
    mean = s / np.maximum(n, 1)
    var = ss / np.maximum(n, 1) - mean ** 2
    return mean, np.sqrt(np.maximum(var, 1e-6))


def prepare(X: np.ndarray, mean: np.ndarray, std: np.ndarray, flag_idx: list[int]) -> np.ndarray:
    miss = np.isnan(X[:, flag_idx]).astype(np.float32)
    Xf = np.where(np.isnan(X), mean[None, :], X)
    Z = (Xf - mean[None, :]) / std[None, :]
    return np.hstack([Z, miss]).astype(np.float32)


# ----------------------------------------------------------------------------
# A tiny numpy MLP with Adam
# ----------------------------------------------------------------------------
@dataclass
class MLPWeights:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def as_list(self):
        return [self.W1, self.b1, self.W2, self.b2]

    @staticmethod
    def from_list(ws):
        return MLPWeights(*[w.copy() for w in ws])

    def copy(self):
        return MLPWeights.from_list(self.as_list())


def init_weights(d_in: int, d_h: int = 32, seed: int = 0) -> MLPWeights:
    rng = np.random.default_rng(seed)
    return MLPWeights(rng.normal(0, np.sqrt(2 / d_in), (d_in, d_h)).astype(np.float32),
                      np.zeros(d_h, np.float32),
                      rng.normal(0, np.sqrt(1 / d_h), (d_h, 1)).astype(np.float32),
                      np.zeros(1, np.float32))


def predict(w: MLPWeights, X: np.ndarray) -> np.ndarray:
    h = np.maximum(X @ w.W1 + w.b1, 0)
    return (h @ w.W2 + w.b2).ravel()


def _grads(w: MLPWeights, X: np.ndarray, y: np.ndarray, delta: float = 0.3):
    z1 = X @ w.W1 + w.b1
    h = np.maximum(z1, 0)
    out = (h @ w.W2 + w.b2).ravel()
    r = out - y
    g_out = np.where(np.abs(r) <= delta, r, delta * np.sign(r)) / len(y)   # Huber loss gradient
    gW2 = h.T @ g_out[:, None]
    gb2 = np.array([g_out.sum()], np.float32)
    gh = g_out[:, None] @ w.W2.T
    gz1 = gh * (z1 > 0)
    gW1 = X.T @ gz1
    gb1 = gz1.sum(0)
    return [gW1, gb1, gW2, gb2]


def train_local(w: MLPWeights, X: np.ndarray, y: np.ndarray, epochs: int = 2, lr: float = 2e-3,
                batch: int = 128, weight_decay: float = 1e-4, seed: int = 0) -> MLPWeights:
    rng = np.random.default_rng(seed)
    params = [p.copy() for p in w.as_list()]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            g = _grads(MLPWeights(*params), X[idx], y[idx])
            t += 1
            for k in range(4):
                g[k] = g[k] + weight_decay * params[k]
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** t)
                vh = v[k] / (1 - b2 ** t)
                params[k] = params[k] - lr * mh / (np.sqrt(vh) + eps)
    return MLPWeights(*params)


def huber(w: MLPWeights, X: np.ndarray, y: np.ndarray, delta: float = 0.3) -> float:
    r = predict(w, X) - y
    a = np.abs(r)
    return float(np.mean(np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))))


# ----------------------------------------------------------------------------
# FedAvg
# ----------------------------------------------------------------------------
def fedavg(global_w: MLPWeights, updates: list[tuple[MLPWeights, int]], clip: float | None = None,
           noise_multiplier: float = 0.0, rng: np.random.Generator | None = None) -> MLPWeights:
    """Weighted average of client models. With `clip`/`noise_multiplier`, applies DP-style
    update clipping and Gaussian noise (central DP-FedAvg)."""
    rng = rng or np.random.default_rng(0)
    gl = global_w.as_list()
    total = sum(n for _, n in updates)
    deltas = []
    for w, n in updates:
        d = [a - b for a, b in zip(w.as_list(), gl)]
        if clip is not None:
            norm = np.sqrt(sum(float((x ** 2).sum()) for x in d))
            scale = min(1.0, clip / max(norm, 1e-12))
            d = [x * scale for x in d]
        deltas.append((d, n))
    if clip is not None and noise_multiplier > 0:
        # equal weights under DP so each client's influence is bounded
        k = len(deltas)
        avg = [sum(d[i] for d, _ in deltas) / k for i in range(4)]
        sigma = noise_multiplier * clip / k
        avg = [a + rng.normal(0, sigma, a.shape).astype(np.float32) for a in avg]
    else:
        avg = [sum(d[i] * (n / total) for d, n in deltas) for i in range(4)]
    return MLPWeights(*[g + a for g, a in zip(gl, avg)])


def run_fedavg(clients: dict[str, tuple[np.ndarray, np.ndarray]], rounds: int = 30, local_epochs: int = 2,
               d_hidden: int = 32, lr: float = 2e-3, client_fraction: float = 1.0, clip: float | None = None,
               noise_multiplier: float = 0.0, seed: int = 0, eval_sets: dict | None = None,
               init: MLPWeights | None = None, callback=None) -> tuple[MLPWeights, list[dict]]:
    """clients: name -> (X_prepared, y). Returns the global model and a per-round history."""
    rng = np.random.default_rng(seed)
    d_in = next(iter(clients.values()))[0].shape[1]
    w = init.copy() if init is not None else init_weights(d_in, d_hidden, seed)
    names = list(clients)
    history = []
    for r in range(1, rounds + 1):
        k = max(1, int(round(client_fraction * len(names))))
        chosen = list(rng.choice(names, size=k, replace=False))
        updates = []
        for i, c in enumerate(chosen):
            X, y = clients[c]
            wc = train_local(w, X, y, epochs=local_epochs, lr=lr, seed=seed * 1000 + r * 100 + i)
            updates.append((wc, len(y)))
        w = fedavg(w, updates, clip=clip, noise_multiplier=noise_multiplier, rng=rng)
        rec = {"round": r, "clients": len(chosen)}
        if eval_sets:
            losses = [huber(w, Xe, ye) * len(ye) for Xe, ye in eval_sets.values()]
            rec["eval_loss"] = float(sum(losses) / sum(len(ye) for _, ye in eval_sets.values()))
        history.append(rec)
        if callback:
            callback(rec, w)
    return w, history


def train_local_only(X: np.ndarray, y: np.ndarray, epochs: int, d_hidden: int = 32, lr: float = 2e-3,
                     seed: int = 0) -> MLPWeights:
    w = init_weights(X.shape[1], d_hidden, seed)
    return train_local(w, X, y, epochs=epochs, lr=lr, seed=seed)


def weights_to_npz_dict(w: MLPWeights, mean: np.ndarray, std: np.ndarray) -> dict:
    return {"W1": w.W1, "b1": w.b1, "W2": w.W2, "b2": w.b2, "mean": mean, "std": std}
