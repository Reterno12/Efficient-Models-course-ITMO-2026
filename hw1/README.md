# ДЗ-1: Analytical performance model небольшой CNN

Здесь находятся модель, аналитические уравнения и [ноутбук](hw1.ipynb), который измеряет все 132 конфигурации на одной CUDA GPU, калибрует параметры только на train points и проверяет предсказания на unseen validation points.

## Запуск

Локально из корня репозитория:

```bash
uv sync --frozen
uv run --frozen jupyter lab
```

Откройте `hw1/hw1.ipynb` и выполните ячейки сверху вниз. Для **Google Colab с T4** достаточно загрузить один файл `hw1.ipynb` через File → Upload notebook (или в расширении VS Code). Модель, equations и measurement встроены в notebook: удалённой среде не нужны локальные `.py` файлы. Первая ячейка проверяет T4 и при необходимости устанавливает `nvidia-ml-py`/`scipy`. Результаты полного запуска на Tesla T4 находятся в `hw1/results/`. Для проверки перед запуском grid: `uv run --frozen python -m unittest discover -s hw1/tests -v`.

`measure_grid(..., resume=True)` дописывает `results/measurements.csv` после каждой точки и пропускает уже измеренные `(S,B)` при повторном запуске. Для чистого измерения на другой GPU удалите старый CSV или используйте отдельный checkout. Не смешивайте результаты разных устройств в одном CSV.

## Соглашения

- Сеть: шесть `Conv–BatchNorm–ReLU` блоков, `MaxPool` после первого, `GlobalAvgPool` и два Linear слоя; `eval()`, FP32, случайные входные тензоры.
- FLOPs: 2 FLOPs на MAC, 2 на output BatchNorm, один на вход GlobalAvgPool; сравнения ReLU/MaxPool исключены. PyTorch profiler сообщает только собственную оценку Conv/Linear FLOPs.
- Peak memory: `torch.cuda.max_memory_allocated()` за forward под `inference_mode()`, включая модель и живой input. Аналитическая формула учитывает пик живых тензоров первого BatchNorm, но не cuDNN workspace, CUDA context и allocator rounding.
- Latency: медиана synchronized wall-clock времени семи forward после трёх warmup. `torch.backends.cudnn.benchmark=False`; оба `allow_tf32=False`.
- Energy: средняя NVML power за серию forward, умноженная на длительность серии и разделённая на число forward. Это энергия **всей GPU**, а не отдельного CUDA kernel. Короткие измерения чувствительны к частоте обновления датчика и внешней нагрузке.
- Validation: 20% пар `(S,B)` выбираются заранее фиксированным seed 2026; их значения не участвуют в calibration. `OOM` фиксируется как отдельный статус.

## Файлы и результаты

- `models.py` — точная сеть.
- `equations.py` — `flops`, `memory`, `latency`, `energy` и вспомогательная `bytes_moved`; все принимают NumPy arrays с broadcasting.
- `measure.py` — полный grid, GPU измерения и обработка `OutOfMemoryError`.
- `calibrate.py` — fit параметров на train и расчёт ошибок.
- `hw1.ipynb` — вывод формул, запуск, calibration, validation и графики.
- `results/measurements.csv`, `results/theta.json`, `results/figures/*.png`, `results/results_summary.md` — реальные измерения, графики и автоматически сформированная сводка, создаваемые notebook.

## Результаты и обсуждение

Измерения выполнены на **Tesla T4** с Python 3.13.15, PyTorch 2.11.0+cu128 и CUDA 12.8. Версия драйвера в сохранённых результатах не указана. Все 132 уникальные пары `(S, B)` завершились успешно; 106 использованы для calibration, 26 заранее отложены для unseen validation. Фактических OOM нет; по сравнению аналитической памяти с ёмкостью GPU предсказанных OOM также нет. Поэтому эта серия не проверяет точность границы OOM.

Медианная абсолютная относительная ошибка на **validation**: FLOPs против оценки PyTorch profiler — **0,25%**, peak memory — **28,90%**, latency — **27,83%**, energy — **21,00%**. В [results_summary.md](results/results_summary.md) сохранена автоматическая сводка, а графики measured vs predicted находятся в [results/figures/](results/figures/).

Модель систематически **недооценивает peak memory**: медианное смещение на validation составляет −28,90%. Аналитическая формула учитывает параметры, вход и одновременно живые активации, но не workspace cuDNN и дополнительные выделения PyTorch. Максимальная ошибка памяти среди validation достигает 78,87% при `S=112, B=1`. Профилировщик FLOPs использует собственную оценку Conv/Linear; наша формула дополнительно считает BatchNorm и GlobalAvgPool, поэтому небольшое устойчивое расхождение ожидаемо.

По наибольшему члену fitted latency model 30 точек отнесены к `launch-bound`, одна (`S=352, B=1`) — к `memory-bound`, 101 — к `compute-bound`. Это **классификация модели**, а не отдельное аппаратное измерение bottleneck. При единственной `memory-bound` точке параметр эффективной пропускной способности плохо определяется данными. Простая формула `launch + max(compute, traffic)` не учитывает выбор разных cuDNN kernels, отдельные запуски операций и изменение частот: медианная ошибка latency на validation 27,83%, а наибольшая — 81,78% при `S=400, B=1`.

Для энергии медианная ошибка на validation 21,00%. Подгонка дала нулевой коэффициент `joules_per_gbyte`, поэтому данные этой серии не позволяют надёжно выделить отдельную цену перемещения байтов. Энергия оценивалась по NVML за серию проходов; частота обновления датчика и внешняя нагрузка GPU добавляют шум. Параметры latency и energy относятся только к этой T4 и требуют новой calibration на другом GPU.

Для сдачи отдельно требуется **собственноручно написанный** `hw1_handwritten.pdf` с выводом четырёх формул и assumptions. Этот файл notebook не создаёт.
