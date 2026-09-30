import numpy as np

from vayusetu.models import federated as FL


def _clients(n=4, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(0, 1, 6)
    out = {}
    for i in range(n):
        X = rng.normal(0, 1, (300, 6)).astype(np.float32)
        y = (X @ w_true * 0.2 + rng.normal(0, 0.05, 300)).astype(np.float32)
        out[f"c{i}"] = (X, y)
    return out


def test_fedavg_learns():
    cl = _clients()
    w0 = FL.init_weights(6, 16, 0)
    before = np.mean([FL.huber(w0, X, y) for X, y in cl.values()])
    w, hist = FL.run_fedavg(cl, rounds=10, local_epochs=2, d_hidden=16, eval_sets=cl)
    after = np.mean([FL.huber(w, X, y) for X, y in cl.values()])
    assert after < before * 0.3
    assert hist[-1]["eval_loss"] < hist[0]["eval_loss"]


def test_federated_stats_match_pooled():
    cl = _clients()
    m, s = FL.combine_stats([FL.local_stats(X.astype(float)) for X, _ in cl.values()])
    pooled = np.vstack([X for X, _ in cl.values()])
    assert np.allclose(m, pooled.mean(0), atol=1e-5) and np.allclose(s, pooled.std(0), atol=1e-4)


def test_dp_path_runs_and_average_of_equal_models_is_identity():
    cl = _clients()
    w, _ = FL.run_fedavg(cl, rounds=3, local_epochs=1, d_hidden=8, clip=1.0, noise_multiplier=0.5)
    assert np.isfinite(w.W1).all()
    base = FL.init_weights(6, 8, 1)
    same = FL.fedavg(base, [(base.copy(), 10), (base.copy(), 30)])
    assert all(np.allclose(a, b) for a, b in zip(same.as_list(), base.as_list()))
