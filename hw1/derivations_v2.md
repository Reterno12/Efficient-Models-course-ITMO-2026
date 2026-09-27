# Формулы и допущения версии 2

## 1. Модель и соглашения

Вход: `B × 3 × S × S`, где `S` кратен 16. Шесть блоков `Conv → BatchNorm → ReLU(inplace)`, после первого блока `MaxPool`. Затем `GlobalAvgPool → Linear(512,256) → ReLU(inplace) → Linear(256,100)`. Каждая `Conv` имеет `bias=False`, `padding=k//2`; сеть работает в `eval()`, FP32 и `torch.inference_mode()`.

| i | Conv | Выходные каналы | Spatial size |
|---|---|---:|---:|
| 1 | 7×7, stride 2, 3→32 | 32 | S/2, после MaxPool — S/4 |
| 2 | 5×5, 32→64 | 64 | S/4 |
| 3 | 3×3, stride 2, 64→128 | 128 | S/8 |
| 4 | 1×1, 128→256 | 256 | S/8 |
| 5 | 3×3, stride 2, 256→256 | 256 | S/16 |
| 6 | 1×1, 256→512 | 512 | S/16 |

## 2. Вывод аналитических уравнений

Обозначим для каждой свёртки `(c_in,i, c_out,i, k_i, r_i)`, где `r_i` — spatial size из таблицы. `W` — байты параметров и buffers сети; `Q(S,B)` — модельный объём передачи данных из `bytes_moved()`.

**FLOPs:** один MAC = 2 FLOPs. `BatchNorm` в `eval()` считаем как scale + shift (2 FLOPs на output), `GlobalAvgPool` — `(n−1)` сложений + 1 деление (всего `n`), для Linear добавляем bias. ReLU и MaxPool используют сравнения и в floating-point operations не входят:

\[
F(S,B)=B\sum_{i=1}^{6}c_{out,i}r_i^2(2c_{in,i}k_i^2+2)
 +512B(S/16)^2+B(2\cdot512\cdot256+256+2\cdot256\cdot100+100).
\]

**Peak memory:** метрика — `torch.cuda.max_memory_allocated()` после прогрева, включая живые allocations библиотек. Старая оценка `M_live=W+76BS²` пропускала постоянный workspace cuBLAS. Исправленная формула для T4, native allocator, одного stream и default workspace:

\[
M(S,B)=W_{alloc}+W_{BLAS}+76BS^2,\qquad
W_{alloc}=\sum_j512\left\lceil\frac{n_j}{512}\right\rceil=4\,187\,136,
\]
\[
W_{BLAS}=(2\cdot4096+8\cdot16)\cdot1024=8\,519\,680\ \text{bytes}.
\]

Здесь `n_j` — байты отдельного параметра/buffer (включая шесть int64 BN counters); размер каждого выделения округляется до 512 bytes. `76BS²=12BS²+2·32BS²` — постоянный вход и две активации первого BatchNorm. При `S` кратном 16 эти три размера уже кратны 512. `W` без округления равен 4 181 312 bytes и по-прежнему используется в сравнении raw parameter storage.

`W_BLAS` задан в [исходниках PyTorch 2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CublasHandlePool.cpp), округление — в [CUDA allocator](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp). **Ни один коэффициент памяти не подгоняется по CSV.** После прогрева Linear этот workspace остаётся выделенным и входит в измеряемый peak. Формула предполагает отсутствие override `CUBLAS_WORKSPACE_CONFIG`, дополнительных streams и посторонних GPU tensors.

Дополнительные временные workspace cuDNN, отдельные пулы cuBLASLt и особенности повторного использования blocks всё ещё не учтены. Поэтому остаётся недооценка, особенно для отдельных batch/shape. CUDA context и свободные reserved blocks не входят в `max_memory_allocated()` и не добавляются к формуле.

**Bytes moved:** для каждой Conv учитываем один read input, один read weights и один write output; для BN/ReLU — read+write; для MaxPool/GAP/Linear аналогично. Игнорируем cache reuse, fusion и cuDNN workspace. Подробная сумма находится в `equations.py`; она не подгоняется по GPU.

**Latency** (seconds):
\[
T(S,B;\theta)=t_0+\max(F(S,B)/P(B),\ Q(S,B)/R),
\quad P(B)=\begin{cases}P_1,& B\le B_*\\P_2,& B>B_*\end{cases}.
\]
`launch-bound`, `memory-bound`, `compute-bound` — соответственно преобладание `t0`, `Q/R`, `F/P(B)`. Порог `B*` и выбор одного/двух compute regimes определяются 5-fold CV только на train по mean APE. Второй режим принимается при улучшении минимум на 5%; затем коэффициенты подгоняются на всём train. Ограничение `P2 ≤ P1` описывает наблюдаемое замедление больших batch. Это эффективные скорости, не аппаратные константы; причина смены режима требует отдельного GPU-профилирования.

**Energy** (joules, вся GPU):
\[
E(S,B;\theta_E)=(P_{idle}+P_{active})T(S,B;\theta)+e_0+e_FF(S,B)/10^9+e_QQ(S,B)/10^9.
\]
`P_idle` измеряем через NVML. `P_active` — дополнительная мощность GPU во время работы; все остальные коэффициенты неотрицательно подгоняем только по train points. Во время измерений мощность включает и idle часть GPU.