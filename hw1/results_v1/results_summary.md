# Результаты на T4
GPU: Tesla T4; PyTorch 2.11.0+cu128; CUDA 12.8.
Grid: 132 pairs; ok=132; OOM=0.
Train=106; unseen validation=26.
Fitted latency: t0=0.000642819 s, P=2.998e+12 FLOP/s, R=9.552e+10 byte/s.
Fitted idle power: 26.9 W.
Regimes (largest fitted term): {'compute-bound': 101, 'launch-bound': 30, 'memory-bound': 1}.

## Validation errors (median APE)
- FLOPs: 0.25% (n=26)
- Memory: 28.90% (n=26)
- Latency: 27.83% (n=26)
- Energy: 21.00% (n=26)

## Обсуждение
FLOPs profiler учитывает Conv/Linear, а аналитическая формула дополнительно BN/GAP; это ожидаемое систематическое расхождение.
Peak memory зависит также от cuDNN workspace и allocator; формула учитывает только живые FP32 тензоры и weights.
Roofline-style latency не описывает kernel selection, launch overhead отдельных operators и изменение clocks; посмотрите пять худших validation points выше.
NVML energy усреднена за серию forwards; частота sensor и внешняя нагрузка создают шум.
Если memory-bound points отсутствуют, fitted R плохо идентифицируется — не трактуйте его как физическую bandwidth GPU.
Actual OOM=0, predicted OOM=0; сравните ошибочные случаи с workspace и доступной VRAM.
