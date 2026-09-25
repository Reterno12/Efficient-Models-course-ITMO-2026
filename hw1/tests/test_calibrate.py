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


if __name__ == '__main__':
    unittest.main()
