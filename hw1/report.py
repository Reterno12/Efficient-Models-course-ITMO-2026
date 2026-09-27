"""Recalibrate saved measurements and regenerate reports without a CUDA run.

Usage: python -m hw1.report --keep-calibration
Defaults to results_v2; version 1 uses its own embedded notebook code.
The software metadata in theta.json describes the original measurements.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from hw1.calibrate import fit_latency, fit_energy, error_summary
from hw1.equations import flops, memory, memory_live_tensors, latency, energy, latency_terms


def write_report(df, results, theta, baseline=None):
    results = Path(results)
    (results / 'figures').mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df['flops_pred'] = flops(df.S, df.B)
    df['memory_pred'] = memory(df.S, df.B)
    df['latency_pred'] = latency(df.S, df.B, theta['latency'])
    df['energy_pred'] = energy(df.S, df.B, theta['energy'])
    metrics = [('flops_profiler', 'flops_pred', 'FLOPs', 1e9, 'GFLOPs'),
               ('memory_bytes', 'memory_pred', 'Peak memory', 2**30, 'GiB'),
               ('latency_s', 'latency_pred', 'Latency', 1e-3, 'ms'),
               ('energy_j', 'energy_pred', 'Energy', 1., 'J')]
    scores = {}
    theta = dict(theta, memory_model='live tensors + 512-byte model allocation rounding + default pre-Hopper cuBLAS workspace; no fit')
    for split, flag in [('train', False), ('validation', True)]:
        part = df[df.status.eq('ok') & df.is_validation.eq(flag)]
        scores[split] = {title: error_summary(part[measured], part[predicted])
                        for measured, predicted, title, _, _ in metrics}
    valid = df[df.status.eq('ok') & df.is_validation]
    scores['memory_comparison_validation'] = {
        'live_tensors_only': error_summary(valid.memory_bytes, memory_live_tensors(valid.S, valid.B)),
        'with_runtime_workspace': scores['validation']['Peak memory']}
    if baseline is not None:
        part = df[df.status.eq('ok') & df.is_validation]
        scores['baseline_validation'] = {
            'Latency': error_summary(part.latency_s, latency(part.S, part.B, baseline['latency'])),
            'Energy': error_summary(part.energy_j, energy(part.S, part.B, baseline['energy']))}
    good = df[df.status.eq('ok')]
    terms = latency_terms(good.S, good.B, theta['latency'])
    regimes = pd.Series(np.array(['launch-bound', 'memory-bound', 'compute-bound'])[
        np.argmax(np.column_stack(terms), axis=1)]).value_counts().to_dict()
    capacity = theta.get('gpu_memory_bytes')
    actual_oom = df.status.eq('OOM').to_numpy()
    scores['oom'] = {'actual': int(actual_oom.sum()), 'predicted': None}
    if capacity is not None:
        predicted_oom = df.memory_pred.to_numpy() > capacity
        scores['oom'].update(predicted=int(predicted_oom.sum()),
                             false_positive=int((predicted_oom & ~actual_oom).sum()),
                             false_negative=int((~predicted_oom & actual_oom).sum()))
    for measured, predicted, title, factor, unit in metrics:
        fig, axes = plt.subplots(3, 4, figsize=(20, 14))
        for ax, s in zip(axes.flat, sorted(df.S.unique())):
            sub = df[df.S.eq(s)].sort_values('B')
            ax.plot(sub.B, sub[predicted] / factor, label='prediction')
            for flag, marker, label in [(False, 'o', 'measured train'), (True, 'D', 'measured validation')]:
                points = sub[sub.status.eq('ok') & sub.is_validation.eq(flag)]
                ax.scatter(points.B, points[measured] / factor, marker=marker, label=label)
            if measured == 'memory_bytes' and capacity is not None:
                ax.axhline(capacity / factor, color='gray', linestyle=':', label='GPU capacity')
                oom = sub[sub.status.eq('OOM')]
                ax.scatter(oom.B, np.full(len(oom), capacity / factor), marker='x', label='OOM')
            ax.set(xlabel='Batch size B', ylabel=f'{title} [{unit}]', title=f'S={s}')
            ax.set_xscale('log', base=2)
            ax.grid(alpha=0.25)
        for ax in list(axes.flat)[df.S.nunique():]:
            ax.axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(results / 'figures' / f'{measured}.png', dpi=160)
        plt.close(fig)
    t = theta['latency']
    summary = ['# Результаты на T4',
               f"GPU: {theta['gpu_name']}; measurement software: {theta['software']}.",
               f'Grid: {len(df)}; ok={len(good)}; OOM={int(df.status.eq("OOM").sum())}.',
               f'Train={int((~df.is_validation).sum())}; validation={int(df.is_validation.sum())}.',
               '', '## Validation errors', '',
               '| Metric | Median APE, % | Mean APE, % |', '|---|---:|---:|']
    for title, score in scores['validation'].items():
        summary.append(f'| {title} | {score["median_ape_percent"]:.2f} | {score["mean_ape_percent"]:.2f} |')
    before_memory = scores['memory_comparison_validation']['live_tensors_only']
    summary += ['', 'Память без подгонки: median APE '
                f'{before_memory["median_ape_percent"]:.2f}% → '
                f'{scores["validation"]["Peak memory"]["median_ape_percent"]:.2f}%.']
    if baseline is not None:
        summary += ['', 'Сравнение с исходными параметрами (тот же CSV и holdout):']
        for title, score in scores['baseline_validation'].items():
            summary.append(f'- {title}: median APE {score["median_ape_percent"]:.2f}% → '
                           f'{scores["validation"][title]["median_ape_percent"]:.2f}%.')
    summary += ['', '## Калибровка и ограничения',
                f'- t0={t["launch_seconds"]:.7g} s; P={t["compute_flops_per_s"]:.7g} FLOP/s; R={t["memory_bytes_per_s"]:.7g} byte/s.',
                f'- Batch threshold={t.get("batch_threshold")}; large-batch P={t.get("large_batch_compute_flops_per_s")} FLOP/s.',
                '- Порог и выбор модели: 5-fold CV только внутри train, критерий mean APE; минимум 5% относительного улучшения. Подробности в theta.json.',
                f'- Idle={theta["energy"]["idle_watts"]:.3f} W; дополнительная active power={theta["energy"]["active_watts"]:.3f} W.',
                f'- Режимы по максимальному члену уравнения: {regimes}. Это классификация модели, не аппаратный профиль.',
                '- FLOPs включает BN/GAP; profiler — Conv/Linear. Память: округлённые веса/buffers + живые тензоры + постоянный cuBLAS workspace 8,125 MiB. Коэффициенты не подгонялись.',
                '- Допущения памяти: T4/pre-Hopper, native allocator, один stream, без CUBLAS_WORKSPACE_CONFIG override и посторонних CUDA tensors. Дополнительные cuDNN/cuBLASLt workspace остаются вне формулы.',
                '- Размер cuBLAS взят из [исходников PyTorch 2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CublasHandlePool.cpp); 512-byte rounding — из [CUDA allocator](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp).',
                '- Два эффективных compute rate описывают наблюдаемый batch-режим. Причина (kernel selection, clocks и т.п.) не установлена; граница между соседними измеренными B не локализована точно.',
                '- R и энергетические коэффициенты могут быть плохо идентифицируемы. Это эффективные параметры всей сети; не физические спецификации GPU.',
                '- NVML energy измеряется за серию forwards, latency — синхронными одиночными вызовами; это добавляет систематическую погрешность.',
                '- Holdout уже использовался в предыдущем отчёте. Коэффициенты и порог не подгонялись по нему, но для независимой проверки новой гипотезы нужны новые измерения.',
                f'- OOM: {scores["oom"]}. ' + ('Фактических OOM нет; точность границы OOM не проверена.' if not actual_oom.any() else 'Сопоставьте ошибки с workspace и доступной VRAM.')]
    if capacity is None:
        summary.append('- Точная ёмкость исходной GPU не сохранена; при offline-пересчёте граница OOM и линия capacity не восстанавливаются по другой GPU.')
    (results / 'theta.json').write_text(json.dumps(theta, ensure_ascii=False, indent=2) + '\n')
    (results / 'errors.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2) + '\n')
    (results / 'results_summary.md').write_text('\n'.join(summary) + '\n')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results_v2')
    parser.add_argument('--keep-calibration', action='store_true',
                        help='Reuse saved latency/energy coefficients; update predictions only')
    args = parser.parse_args()
    results = args.results
    df = pd.read_csv(results / 'measurements.csv')
    theta = json.loads((results / 'theta.json').read_text())
    if not args.keep_calibration:
        theta['latency'] = fit_latency(df)
        theta['energy'] = fit_energy(df, theta['latency'])
        theta['calibration_software'] = dict(python=platform.python_version(),
                                            numpy=np.__version__, scipy=scipy.__version__)
    baseline_path = results / 'theta_baseline.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    scores = write_report(df, results, theta, baseline)
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
