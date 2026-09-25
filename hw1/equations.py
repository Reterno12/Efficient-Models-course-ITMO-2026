"""Explicit analytic equations. S and B are NumPy-broadcastable scalars/arrays.

Assumptions: NCHW FP32, eval+inference_mode, Conv-BN-ReLU after each conv,
no residuals, inplace ReLU. Arithmetic counts floating add/multiply/divide;
comparisons (ReLU/MaxPool) are not FLOPs. Memory is a lower-bound live-tensor
estimate; cuDNN workspaces, allocator rounding and CUDA context are excluded.
"""
import numpy as np

# input channels, output channels, kernel, output resolution divisor
CONVS = ((3, 32, 7, 2), (32, 64, 5, 4), (64, 128, 3, 8),
         (128, 256, 1, 8), (256, 256, 3, 16), (256, 512, 1, 16))


def _sb(image_size, batch):
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float),
                               np.asarray(batch, dtype=float))
    if np.any(s <= 0) or np.any(b <= 0):
        raise ValueError("S and B must be positive")
    return s, b


def model_bytes():
    """FP32 parameters and BN buffers, including six int64 batch counters."""
    conv_params = sum(ci * co * k * k for ci, co, k, _ in CONVS)
    bn_floats = sum(4 * co for _, co, _, _ in CONVS)
    linear_params = 512 * 256 + 256 + 256 * 100 + 100
    return 4 * (conv_params + bn_floats + linear_params) + 8 * len(CONVS)


def flops(image_size, batch):
    """Forward FLOPs: 2 per MAC, 2 per BN output, 1 per GAP input, biases."""
    s, b = _sb(image_size, batch)
    total = 0.0
    for ci, co, k, divisor in CONVS:
        pixels = (s / divisor) ** 2
        total += b * co * pixels * (2 * ci * k * k + 2)
    total += b * 512 * (s / 16) ** 2  # GAP: (n-1) adds + division
    total += b * (2 * 512 * 256 + 256 + 2 * 256 * 100 + 100)
    return total


def memory(image_size, batch):
    """Predicted peak bytes: weights + persistent input + two conv1 activations.

The input x remains referenced by the caller. BN1 holds both conv1 output
and BN output; both contain B*32*(S/2)^2 float32 values. This is the
largest live pair in this sequential network. It omits temporary workspaces.
"""
    s, b = _sb(image_size, batch)
    return model_bytes() + 4 * b * (3 * s**2 + 2 * 32 * (s / 2)**2)


def bytes_moved(image_size, batch):
    """Approximate global-memory traffic: each op reads inputs and writes outputs.

Cache reuse, temporary workspaces, kernel fusion and allocator traffic are
excluded. Conv weights are counted once per forward, not once per pixel.
"""
    s, b = _sb(image_size, batch)
    total_elements = 0.0
    previous = 3 * b * s**2
    for index, (ci, co, k, divisor) in enumerate(CONVS):
        output = b * co * (s / divisor)**2
        total_elements += previous + ci * co * k * k + output  # conv
        total_elements += 2 * output + 4 * co  # BN read/write + vectors
        total_elements += 2 * output  # inplace ReLU read/write
        previous = output
        if index == 0:
            pooled = b * co * (s / 4)**2
            total_elements += previous + pooled
            previous = pooled
    total_elements += previous + b * 512  # GAP
    total_elements += b * 512 + 512 * 256 + b * 256  # linear1
    total_elements += 2 * b * 256  # inplace ReLU
    total_elements += b * 256 + 256 * 100 + b * 100  # linear2
    return 4 * total_elements


def latency(image_size, batch, theta):
    """Roofline-style seconds: launch overhead + max(compute, memory time)."""
    return (theta["launch_seconds"] + np.maximum(
        flops(image_size, batch) / theta["compute_flops_per_s"],
        bytes_moved(image_size, batch) / theta["memory_bytes_per_s"]))


def energy(image_size, batch, theta_energy):
    """Whole-GPU joules: idle baseline over predicted time + dynamic terms."""
    return (theta_energy["idle_watts"] *
            latency(image_size, batch, theta_energy["latency"]) +
            theta_energy["fixed_joules"] +
            theta_energy["joules_per_gflop"] * flops(image_size, batch) / 1e9 +
            theta_energy["joules_per_gbyte"] * bytes_moved(image_size, batch) / 1e9)
