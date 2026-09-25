"""Fit latency and whole-GPU energy parameters on train configurations only."""
import numpy as np
from scipy.optimize import least_squares, nnls

from hw1.equations import flops, bytes_moved, latency


def _training(df, column):
    train = df[(df["status"] == "ok") & (~df["is_validation"])].copy()
    train = train[np.isfinite(train[column])]
    if len(train) < 6:
        raise ValueError(f"Need at least 6 finite train measurements for {column}")
    return train


def fit_latency(df):
    train = _training(df, "latency_s")
    f = np.asarray(flops(train.S.to_numpy(), train.B.to_numpy()) / 1e9)
    m = np.asarray(bytes_moved(train.S.to_numpy(), train.B.to_numpy()) / 1e9)
    observed = train.latency_s.to_numpy(dtype=float)
    scale = np.maximum(observed, np.median(observed) * 0.05)

    def residual(p):
        return (p[0] + np.maximum(p[1] * f, p[2] * m) - observed) / scale

    # max() changes branch at memory/compute boundaries; try several starts.
    base_a = max(np.median(observed / np.maximum(f, 1e-12)), 1e-8)
    base_c = max(np.median(observed / np.maximum(m, 1e-12)), 1e-8)
    fits = []
    for af in (0.1, 1.0, 10.0):
        for cf in (0.1, 1.0, 10.0):
            fit = least_squares(residual,
                                [max(np.min(observed) * 0.5, 1e-6),
                                 base_a * af, base_c * cf],
                                bounds=(0, np.inf), x_scale="jac", max_nfev=5000)
            if fit.success:
                fits.append(fit)
    if not fits:
        raise RuntimeError("Latency fit failed for all initializations")
    fit = min(fits, key=lambda candidate: np.sum(candidate.fun**2))
    t0, seconds_per_gflop, seconds_per_gbyte = fit.x
    return {"launch_seconds": float(t0),
            "compute_flops_per_s": float(1e9 / max(seconds_per_gflop, 1e-15)),
            "memory_bytes_per_s": float(1e9 / max(seconds_per_gbyte, 1e-15))}


def fit_energy(df, theta_latency):
    train = _training(df, "energy_j")
    idle = float(np.median(train.idle_watts.to_numpy(dtype=float)))
    if not np.isfinite(idle):
        raise ValueError("NVML idle power is unavailable; cannot fit energy")
    s, b = train.S.to_numpy(), train.B.to_numpy()
    x = np.column_stack((np.ones(len(train)), flops(s, b) / 1e9,
                         bytes_moved(s, b) / 1e9))
    target = train.energy_j.to_numpy(dtype=float) - idle * latency(s, b, theta_latency)
    # Weight by observed energy so large, slow configurations do not erase
    # relative accuracy on short forwards. The floor is train-only.
    scale = np.maximum(train.energy_j.to_numpy(dtype=float),
                       np.quantile(train.energy_j, 0.05))
    (fixed, per_gflop, per_gbyte), _ = nnls(x / scale[:, None], target / scale)
    return {"latency": theta_latency, "idle_watts": idle,
            "fixed_joules": float(fixed),
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
