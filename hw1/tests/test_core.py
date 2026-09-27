import unittest
import numpy as np
import torch

from hw1.models import SmallCNN
from hw1.equations import flops, memory, bytes_moved, latency, energy, model_bytes
from hw1.measure import build_grid


class CoreTests(unittest.TestCase):
    def test_model_shape_and_analytic_parameter_bytes(self):
        model = SmallCNN().eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 100))
        actual = sum(p.numel() * p.element_size() for p in model.parameters())
        actual += sum(b.numel() * b.element_size() for b in model.buffers())
        self.assertEqual(model_bytes(), actual)

    def test_equations_broadcast_and_scaling(self):
        s = np.array([32, 64])[:, None]
        b = np.array([1, 2, 4])[None, :]
        for fn in (flops, memory, bytes_moved):
            result = fn(s, b)
            self.assertEqual(result.shape, (2, 3))
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertTrue(np.all(result > 0))
        self.assertTrue(np.isclose(flops(32, 2), 2 * flops(32, 1)))
        self.assertEqual(memory(32, 2) - memory(32, 1), 76 * 32**2)

    def test_memory_includes_rounded_model_and_persistent_blas_workspace(self):
        # Hand calculation for S=32, B=1, native allocator / pre-Hopper CUDA:
        # rounded model 4,187,136 + documented cuBLAS 8,519,680
        # + input 12,288 + two BN1 activations 2*32,768.
        self.assertEqual(memory(32, 1), 12_784_640)
        self.assertEqual(memory(32, 2), 12_862_464)

    def test_predictions_accept_arrays(self):
        theta = {"launch_seconds": 1e-4, "compute_flops_per_s": 1e12,
                 "memory_bytes_per_s": 1e11}
        theta_energy = {"latency": theta, "idle_watts": 10.0,
                        "fixed_joules": 0.001, "joules_per_gflop": 0.01,
                        "joules_per_gbyte": 0.02}
        self.assertEqual(latency(np.array([32, 64]), 2, theta).shape, (2,))
        self.assertEqual(energy(np.array([32, 64]), 2, theta_energy).shape, (2,))

    def test_grid_is_reproducible_and_has_unseen_validation_pairs(self):
        a = build_grid(seed=2026)
        b = build_grid(seed=2026)
        self.assertTrue(a.equals(b))
        self.assertEqual(len(a), 132)
        self.assertEqual(int(a.duplicated(["S", "B"]).sum()), 0)
        self.assertTrue(a.is_validation.any() and (~a.is_validation).any())
        self.assertTrue(set(a.S) >= {32, 64, 128, 224, 256, 384, 512})
        self.assertTrue(set(a.B) >= {1, 2, 4, 8, 16, 32, 64, 128, 256})


if __name__ == '__main__':
    unittest.main()
