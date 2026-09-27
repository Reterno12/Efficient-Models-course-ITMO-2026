# Результаты на T4
GPU: Tesla T4; measurement software: {'python': '3.13.15', 'torch': '2.11.0+cu128', 'cuda': '12.8'}.
Grid: 132; ok=132; OOM=0.
Train=106; validation=26.

## Validation errors

| Metric | Median APE, % | Mean APE, % |
|---|---:|---:|
| FLOPs | 0.25 | 0.25 |
| Peak memory | 22.03 | 21.90 |
| Latency | 14.63 | 16.21 |
| Energy | 6.89 | 11.13 |

Память без подгонки: median APE 28.90% → 22.03%.

## Калибровка и ограничения
- t0=0.0007322803 s; P=3.715082e+12 FLOP/s; R=1.199993e+11 byte/s.
- Batch threshold=64; large-batch P=1939449651515.796 FLOP/s.
- Порог и выбор модели: 5-fold CV только внутри train, критерий mean APE; минимум 5% относительного улучшения. Подробности в theta.json.
- Idle=30.970 W; дополнительная active power=27.783 W.
- Режимы по максимальному члену уравнения: {'compute-bound': 98, 'launch-bound': 34}. Это классификация модели, не аппаратный профиль.
- FLOPs включает BN/GAP; profiler — Conv/Linear. Память: округлённые веса/buffers + живые тензоры + постоянный cuBLAS workspace 8,125 MiB. Коэффициенты не подгонялись.
- Допущения памяти: T4/pre-Hopper, native allocator, один stream, без CUBLAS_WORKSPACE_CONFIG override и посторонних CUDA tensors. Дополнительные cuDNN/cuBLASLt workspace остаются вне формулы.
- Размер cuBLAS взят из [исходников PyTorch 2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CublasHandlePool.cpp); 512-byte rounding — из [CUDA allocator](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp).
- Два эффективных compute rate описывают наблюдаемый batch-режим. Причина (kernel selection, clocks и т.п.) не установлена; граница между соседними измеренными B не локализована точно.
- R и энергетические коэффициенты могут быть плохо идентифицируемы. Это эффективные параметры всей сети; не физические спецификации GPU.
- NVML energy измеряется за серию forwards, latency — синхронными одиночными вызовами; это добавляет систематическую погрешность.
- Holdout уже использовался в предыдущем отчёте. Коэффициенты и порог не подгонялись по нему, но для независимой проверки новой гипотезы нужны новые измерения.
- OOM: {'actual': 0, 'predicted': 0, 'false_positive': 0, 'false_negative': 0}. Фактических OOM нет; точность границы OOM не проверена.
