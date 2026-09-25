# Результаты на Tesla T4

Эта папка содержит распакованный `hw1_results.zip`, полученный после полного запуска `hw1.ipynb` на Tesla T4:

- `measurements.csv` — все 132 пары `(S, B)`, измерения и флаг validation;
- `theta.json` — подобранные параметры latency и energy, версии Python/PyTorch/CUDA;
- `results_summary.md` — автоматически сформированная сводка;
- `figures/` — четыре графика measured vs predicted.

Краткое обсуждение результатов находится в [основном README](../README.md). Данные и параметры этой GPU не следует объединять с результатами другой GPU без повторных измерений и calibration.
