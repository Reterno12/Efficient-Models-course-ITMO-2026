# Результаты версии 2

Текущий прогон: `20260927_090211_ab1cd7d2`, источник `hw1_results_20260927_090211_ab1cd7d2.zip`. Все 132 конфигурации Tesla T4 завершились успешно; 106 train, 26 validation.

CSV и коэффициенты latency/energy сохранены без изменений. В архиве отчёт памяти ещё использовал старую формулу; здесь ошибки и графики пересчитаны актуальными `.py` версии 2. Архивные CSV/theta/ошибки/сводка сохранены в `source_run/`, происхождение — в `import_manifest.json`.

Median APE: FLOPs **0,25%**, Memory **22,03%**, Latency **14,63%**, Energy **6,89%**. Сводка: [results_summary.md](results_summary.md). Сравнение двух версий: [versions_comparison.json](../versions_comparison.json).

Из корня репозитория: `uv run --frozen python -m hw1.report --keep-calibration`. Новые GPU-прогоны notebook сохраняет в `results_v2/runs/<RUN_ID>/`.
