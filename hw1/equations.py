"""Explicit analytic equations. S and B are NumPy-broadcastable scalars/arrays.

Assumptions: NCHW FP32, eval+inference_mode, Conv-BN-ReLU after each conv,
no residuals, inplace ReLU. Arithmetic counts floating add/multiply/divide;
comparisons (ReLU/MaxPool) are not FLOPs. Memory includes live tensors,
512-byte native-allocator rounding of model storage and one persistent
pre-Hopper cuBLAS workspace after warmup. Additional cuDNN/cuBLASLt
workspaces and allocator block reuse are excluded; CUDA context is not
part of torch.cuda.max_memory_allocated().
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


def model_allocated_bytes():
    """Model storage with PyTorch native allocator's 512-byte granularity.

    Each parameter/buffer has its own allocation, including six 8-byte BN
    counters. This is derived from tensor shapes, not fitted to measurements.
    """
    sizes = []
    for ci, co, k, _ in CONVS:
        sizes.extend([4 * ci * co * k * k, *([4 * co] * 4), 8])
    sizes.extend([4 * 512 * 256, 4 * 256, 4 * 256 * 100, 4 * 100])
    return sum(512 * ((size + 511) // 512) for size in sizes)


def memory_live_tensors(image_size, batch):
    """Original ideal tensor-storage estimate, retained as a comparison."""
    s, b = _sb(image_size, batch)
    return model_bytes() + 76 * b * s**2


def memory(image_size, batch):
    """Peak estimate for warmed-up T4, FP32, one stream, native allocator.

    M = W_alloc + W_cuBLAS + input + two BN1 activations.
    W_cuBLAS = (2*4096 + 8*16) KiB is the PyTorch pre-Hopper default,
    allocated persistently through the CUDA caching allocator after Linear.
    Source: pytorch v2.11.0/aten/src/ATen/cuda/CublasHandlePool.cpp,
    parseChosenWorkspaceSize()/getNewWorkspace(). It is NOT a fitted offset.

    Assumes no CUBLAS_WORKSPACE_CONFIG override, no extra live CUDA tensors,
    and S a multiple of 16 (input/BN1 sizes are already multiples of 512).
    cuDNN scratch space and additional cuBLASLt pools remain unmodelled;
    the prediction is a baseline, not an exact peak or an OOM guarantee.
    """
    s, b = _sb(image_size, batch)
    cublas_workspace = (2 * 4096 + 8 * 16) * 1024
    return model_allocated_bytes() + cublas_workspace + 76 * b * s**2


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


def latency_terms(image_size, batch, theta):
    """Launch, memory and compute times; optional train-selected batch regime.

    Two effective compute rates approximate a change of convolution execution
    regime. They are empirical rates, not GPU hardware peak specifications.
    Old three-parameter dictionaries retain their original interpretation.
    """
    s, b = _sb(image_size, batch)
    rate = theta["compute_flops_per_s"]
    if theta.get("batch_threshold") is not None:
        rate = np.where(b > theta["batch_threshold"],
                        theta["large_batch_compute_flops_per_s"], rate)
    return (np.full(s.shape, theta["launch_seconds"]),
            bytes_moved(s, b) / theta["memory_bytes_per_s"],
            flops(s, b) / rate)


def latency(image_size, batch, theta):
    """Seconds: t0 + max(F/P(B), Q/R), with a train-calibrated P(B)."""
    launch, traffic, compute = latency_terms(image_size, batch, theta)
    return launch + np.maximum(compute, traffic)


def energy(image_size, batch, theta_energy):
    """Whole-GPU joules: idle + active power over predicted time + work terms."""
    return ((theta_energy["idle_watts"] + theta_energy.get("active_watts", 0.0)) *
            latency(image_size, batch, theta_energy["latency"]) +
            theta_energy["fixed_joules"] +
            theta_energy["joules_per_gflop"] * flops(image_size, batch) / 1e9 +
            theta_energy["joules_per_gbyte"] * bytes_moved(image_size, batch) / 1e9)
