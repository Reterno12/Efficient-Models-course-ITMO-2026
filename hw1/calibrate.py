"""Fit latency and whole-GPU energy parameters on train configurations only."""
import numpy as np
from scipy.optimize import least_squares, nnls

from hw1.equations import flops, bytes_moved, latency


def _training(df, column):
    train = df[(df["status"] == "ok") & (~df["is_validation"])].copy()
    train = train[np.isfinite(train[column]) & (train[column] > 0)]
    if len(train) < 6:
        raise ValueError(f"Need at least 6 finite train measurements for {column}")
    return train


def _fit_latency_model(train, threshold=None):
    f = np.asarray(flops(train.S.to_numpy(), train.B.to_numpy()) / 1e9)
    m = np.asarray(bytes_moved(train.S.to_numpy(), train.B.to_numpy()) / 1e9)
    observed = train.latency_s.to_numpy(dtype=float)
    scale = np.maximum(observed, np.median(observed) * 0.05)
    large = train.B.to_numpy() > threshold if threshold is not None else None

    def residual(p):
        compute = f * (p[1] + p[3] * large) if large is not None else f * p[1]
        return (p[0] + np.maximum(compute, p[2] * m) - observed) / scale

    base_a = max(np.median(observed / np.maximum(f, 1e-12)), 1e-8)
    base_c = max(np.median(observed / np.maximum(m, 1e-12)), 1e-8)
    fits = []
    for af in (0.1, 1.0, 10.0):
        for cf in (0.1, 1.0, 10.0):
            initial = [max(np.min(observed) * 0.5, 1e-6), base_a * af, base_c * cf]
            if large is not None:
                initial.append(base_a)
            fit = least_squares(residual, initial, bounds=(0, np.inf),
                                x_scale="jac", max_nfev=2000)
            if fit.success:
                fits.append(fit)
    if not fits:
        raise RuntimeError("Latency fit failed for all initializations")
    fit = min(fits, key=lambda candidate: np.sum(candidate.fun**2))
    t0, a, c = fit.x[:3]
    theta = {"launch_seconds": float(t0),
             "compute_flops_per_s": float(1e9 / max(a, 1e-15)),
             "memory_bytes_per_s": float(1e9 / max(c, 1e-15))}
    if threshold is not None:
        theta.update(batch_threshold=int(threshold),
                     large_batch_compute_flops_per_s=float(1e9 / max(a + fit.x[3], 1e-15)))
    return theta


def fit_latency(df):
    """Select one/two compute regimes by five-fold CV entirely inside train.

    Candidate thresholds are observed train batch sizes with >=20% of train
    on either side. A second regime must reduce CV mean APE by at least 5%
    relative to the baseline; the selected model is then fitted on all train.
    No validation coordinates, targets or statistics enter model selection.
    """
    train = _training(df, "latency_s").sort_values(["S", "B"]).reset_index(drop=True)
    thresholds = [None] + [int(b) for b in sorted(train.B.unique())
                           if 0.2 <= np.mean(train.B > b) <= 0.8]
    folds = np.random.default_rng(2026).permutation(len(train)) % 5
    scores = []
    for threshold in thresholds:
        predicted = np.empty(len(train))
        for fold in range(5):
            mask = folds == fold
            theta = _fit_latency_model(train.loc[~mask], threshold)
            predicted[mask] = latency(train.loc[mask, "S"].to_numpy(),
                                      train.loc[mask, "B"].to_numpy(), theta)
        scores.append(error_summary(train.latency_s, predicted))
    best = min(range(len(scores)), key=lambda i: scores[i]["mean_ape_percent"])
    if scores[best]["mean_ape_percent"] >= 0.95 * scores[0]["mean_ape_percent"]:
        best = 0
    theta = _fit_latency_model(train, thresholds[best])
    theta["selection"] = {"method": "5-fold train-only CV; mean APE; 5% relative improvement required",
                          "seed": 2026, "n_train": len(train),
                          "candidates": [dict(batch_threshold=t, **score)
                                         for t, score in zip(thresholds, scores)]}
    return theta


def fit_energy(df, theta_latency):
    train = _training(df, "energy_j")
    idle = float(np.median(train.idle_watts.to_numpy(dtype=float)))
    if not np.isfinite(idle):
        raise ValueError("NVML idle power is unavailable; cannot fit energy")
    s, b = train.S.to_numpy(), train.B.to_numpy()
    predicted_time = latency(s, b, theta_latency)
    x = np.column_stack((np.ones(len(train)), flops(s, b) / 1e9,
                         bytes_moved(s, b) / 1e9, predicted_time))
    target = train.energy_j.to_numpy(dtype=float) - idle * predicted_time
    # Weight by observed energy so large, slow configurations do not erase
    # relative accuracy on short forwards. The floor is train-only.
    scale = np.maximum(train.energy_j.to_numpy(dtype=float),
                       np.quantile(train.energy_j, 0.05))
    (fixed, per_gflop, per_gbyte, active), _ = nnls(x / scale[:, None], target / scale)
    return {"latency": theta_latency, "idle_watts": idle,
            "active_watts": float(active), "fixed_joules": float(fixed),
            "joules_per_gflop": float(per_gflop),
            "joules_per_gbyte": float(per_gbyte)}


def error_summary(measured, predicted):
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(predicted) & (measured > 0)
    if not np.any(valid):
        return {"n": 0, "median_ape_percent": None, "mean_ape_percent": None}
    ape = 100 * np.abs(predicted[valid] - measured[valid]) / measured[valid]
    return {"n": int(valid.sum()), "median_ape_percent": float(np.median(ape)),
            "mean_ape_percent": float(np.mean(ape))}
