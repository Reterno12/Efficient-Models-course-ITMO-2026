"""GPU measurements for the 132-point grid; safe to resume after OOM/interruption."""
import gc
import math
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hw1.models import SmallCNN

ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", torch.cuda.OutOfMemoryError)

BASE_S = (32, 64, 128, 224, 256, 384, 512)
BASE_B = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def build_grid(seed=2026):
    rng = np.random.default_rng(seed)
    extra_s = rng.choice([v for v in range(32, 513, 16) if v not in BASE_S],
                         size=4, replace=False)
    extra_b = rng.choice([v for v in range(1, 257) if v not in BASE_B],
                         size=3, replace=False)
    points = [(int(s), int(b)) for s in sorted((*BASE_S, *extra_s))
              for b in sorted((*BASE_B, *extra_b))]
    flags = np.zeros(len(points), dtype=bool)
    flags[rng.choice(len(points), size=round(0.2 * len(points)), replace=False)] = True
    return pd.DataFrame({"S": [p[0] for p in points],
                         "B": [p[1] for p in points],
                         "is_validation": flags})


def _synchronize():
    torch.cuda.synchronize()


def _one_forward(model, x):
    with torch.inference_mode():
        return model(x)


def _power_series(handle, stop, samples):
    import pynvml
    while not stop.is_set():
        try:
            samples.append((time.perf_counter(),
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0))
        except pynvml.NVMLError:
            break
        stop.wait(0.02)


def _energy_per_forward(model, x, repetitions, handle):
    if handle is None:
        return float("nan")
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=_power_series, args=(handle, stop, samples),
                               daemon=True)
    _synchronize()
    sampler.start()
    start = time.perf_counter()
    for _ in range(repetitions):
        _one_forward(model, x)
    _synchronize()
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join()
    if len(samples) < 2:
        return float("nan")
    # Sensor samples cover the whole-GPU power during repeated forwards.
    return float(np.mean([watts for _, watts in samples]) * elapsed / repetitions)


def measure_one(model, image_size, batch, handle=None, latency_repeats=7):
    """Median synchronized wall time, peak PyTorch allocated bytes, GPU joules.

    Input allocation is included in peak, as the input stays alive throughout
    forward. The returned output is released before the memory reading.
    """
    x = torch.randn(batch, 3, image_size, image_size, device="cuda", dtype=torch.float32)
    try:
        for _ in range(3):
            _one_forward(model, x)
        _synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(latency_repeats):
            _synchronize()
            start = time.perf_counter()
            _one_forward(model, x)
            _synchronize()
            times.append(time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated()
        median = float(np.median(times))
        energy_repeats = max(5, min(1000, math.ceil(0.5 / median)))
        joules = _energy_per_forward(model, x, energy_repeats, handle)
        # Profiler reports estimated conv/linear FLOPs, not a hardware counter.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True, record_shapes=True,
        ) as prof:
            _one_forward(model, x)
            _synchronize()
        prof_flops = sum(event.flops for event in prof.key_averages())
        return {"status": "ok", "latency_s": median,
                "memory_bytes": int(peak), "energy_j": joules,
                "flops_profiler": int(prof_flops),
                "energy_repeats": energy_repeats}
    finally:
        del x


def init_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except Exception as exc:
        print(f"NVML unavailable: {exc}; energy_j will be NaN")
        return None


def idle_power_watts(handle, samples=20):
    if handle is None:
        return float("nan")
    import pynvml
    _synchronize()
    values = []
    for _ in range(samples):
        values.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        time.sleep(0.025)
    return float(np.median(values))


def measure_grid(path, seed=2026, resume=True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for real measurements")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device_name = torch.cuda.get_device_name(0)
    model = SmallCNN().cuda().eval()
    handle = init_nvml()
    idle = idle_power_watts(handle)
    grid = build_grid(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = set()
    if resume and path.exists() and path.stat().st_size:
        rows = pd.read_csv(path).to_dict("records")
        prior_devices = {str(r.get("gpu_name")) for r in rows}
        if prior_devices != {device_name}:
            raise ValueError(f"Existing CSV is from {prior_devices}; current GPU is {device_name}")
        done = {(int(r["S"]), int(r["B"])) for r in rows}
    for index, row in grid.iterrows():
        s, b = int(row.S), int(row.B)
        if (s, b) in done:
            continue
        record = {"S": s, "B": b, "is_validation": bool(row.is_validation),
                  "gpu_name": device_name, "idle_watts": idle}
        was_oom = False
        try:
            record.update(measure_one(model, s, b, handle))
        except (torch.cuda.OutOfMemoryError, ACCELERATOR_ERROR) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            record.update(status="OOM", latency_s=float("nan"),
                          memory_bytes=float("nan"), energy_j=float("nan"),
                          flops_profiler=float("nan"), energy_repeats=0)
            was_oom = True
        if was_oom:
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(record)
        pd.DataFrame(rows).to_csv(path, index=False)
        if (index + 1) % 10 == 0 or record["status"] == "OOM":
            print(f"{index+1}/{len(grid)}: S={s}, B={b}, {record['status']}")
    return pd.DataFrame(rows).sort_values(["S", "B"]).reset_index(drop=True)
