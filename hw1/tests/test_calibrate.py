import unittest
import numpy as np
import pandas as pd

from hw1.equations import latency, energy
from hw1.calibrate import fit_latency, fit_energy


class CalibrationTests(unittest.TestCase):
    def test_fit_uses_only_training_points(self):
        s = np.repeat([32, 64, 128, 256, 512], 9)
        b = np.tile([1, 2, 4, 8, 16, 32, 64, 128, 256], 5)
        true_t = {"launch_seconds": 2e-4, "compute_flops_per_s": 1e12,
                  "memory_bytes_per_s": 1e11}
        true_e = {"latency": true_t, "idle_watts": 12.0,
                  "fixed_joules": 0.002, "joules_per_gflop": 0.005,
                  "joules_per_gbyte": 0.01}
        times = latency(s, b, true_t)
        joules = energy(s, b, true_e)
        df = pd.DataFrame({"S": s, "B": b, "status": "ok",
                           "is_validation": np.arange(len(s)) % 5 == 0,
                           "latency_s": times, "energy_j": joules,
                           "idle_watts": 12.0})
        df.loc[df.is_validation, ["latency_s", "energy_j"]] *= 100
        theta_t = fit_latency(df)
        theta_e = fit_energy(df, theta_t)
        training = df[~df.is_validation]
        np.testing.assert_allclose(latency(training.S, training.B, theta_t),
                                   times[~df.is_validation], rtol=0.02)
        np.testing.assert_allclose(energy(training.S, training.B, theta_e),
                                   joules[~df.is_validation], rtol=0.02)

    def test_two_compute_regimes_generalize_without_validation_leakage(self):
        s = np.repeat([32, 64, 128, 256, 512], 9)
        b = np.tile([1, 2, 4, 8, 16, 32, 64, 128, 256], 5)
        from hw1.equations import flops, bytes_moved
        times = 0.0003 + np.maximum(
            flops(s, b) / np.where(b > 64, 1e12, 3e12),
            bytes_moved(s, b) / 1e11)
        df = pd.DataFrame(dict(S=s, B=b, status="ok", latency_s=times,
                               is_validation=np.arange(len(s)) % 5 == 0))
        fitted = fit_latency(df)
        np.testing.assert_allclose(latency(s, b, fitted), times, rtol=0.03)
        poisoned = df.copy()
        poisoned.loc[poisoned.is_validation, ["S", "B", "latency_s"]] *= 100
        self.assertEqual(fitted, fit_latency(poisoned))

    def test_energy_recovers_active_power(self):
        from hw1.equations import flops, bytes_moved
        s = np.repeat([32, 64, 128, 256, 512], 9)
        b = np.tile([1, 2, 4, 8, 16, 32, 64, 128, 256], 5)
        theta = dict(launch_seconds=0.0003, compute_flops_per_s=3e12,
                     memory_bytes_per_s=1e11, batch_threshold=64,
                     large_batch_compute_flops_per_s=1e12)
        times = 0.0003 + np.maximum(flops(s, b) / np.where(b > 64, 1e12, 3e12),
                                    bytes_moved(s, b) / 1e11)
        joules = 60 * times
        df = pd.DataFrame(dict(S=s, B=b, status="ok", latency_s=times,
                               energy_j=joules, idle_watts=20.,
                               is_validation=np.arange(len(s)) % 5 == 0))
        fitted = fit_energy(df, theta)
        np.testing.assert_allclose(energy(s, b, fitted), joules, rtol=0.02)
        poisoned = df.copy()
        poisoned.loc[poisoned.is_validation, ["energy_j", "idle_watts"]] *= 100
        self.assertEqual(fitted, fit_energy(poisoned, theta))


if __name__ == '__main__':
    unittest.main()
