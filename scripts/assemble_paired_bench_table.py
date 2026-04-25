"""Assemble a unified paired-variant benchmark comparison across Gemma-3 + Gemma-4.

Reads per-model `summary.csv` files produced by run_genuine_benchmarks.py
and emits a single table keyed by (task, metric) with columns for every
(model, variant). Computes per-variant deltas vs. the base for easy reading.

Usage:
    python scripts/assemble_paired_bench_table.py \\
        --gemma3-bench D:/research/benchmarks/gemma3-brain-bench-<ts> \\
        --gemma4-bench D:/research/benchmarks/gemma4-brain-bench-<ts> \\
        [--gemma4-adapters-bench D:/research/benchmarks/gemma4-adapters-<ts>] \\
        --out D:/research/benchmarks/paired_<ts>.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_METRICS = {
    'arc_challenge':     'acc,none',
    'openbookqa':        'acc_norm,none',
    'piqa':              'acc,none',
    'truthfulqa_mc1':    'acc,none',
    'gsm8k':             'exact_match,strict-match',
}


def _load(bench_dir: Path) -> dict[tuple[str, str], float]:
    """Return {(variant, task): primary_metric_value} from a summary.csv."""
    out: dict[tuple[str, str], float] = {}
    summ = bench_dir / 'summary.csv'
    if not summ.exists():
        return out
    with summ.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            variant = row.get('variant') or ''
            task = row.get('task') or ''
            metric_key = KEY_METRICS.get(task)
            if not metric_key:
                continue
            val = row.get(metric_key)
            if val in (None, '', 'None'):
                continue
            try:
                out[(variant, task)] = float(val)
            except ValueError:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gemma3-bench', type=Path, required=True)
    ap.add_argument('--gemma4-bench', type=Path, required=True)
    ap.add_argument('--gemma4-adapters-bench', type=Path, default=None,
                    help='optional second gemma4 run containing just the adapter variants')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    g3 = _load(args.gemma3_bench)
    g4 = _load(args.gemma4_bench)
    if args.gemma4_adapters_bench:
        g4.update(_load(args.gemma4_adapters_bench))

    tasks = sorted({t for (_, t) in set(g3) | set(g4)})

    header = ['task', 'metric',
              'g3_base', 'g3_v2', 'g3_v2_delta', 'g3_v3', 'g3_v3_delta',
              'g4_base', 'g4_r32', 'g4_r32_delta', 'g4_r64', 'g4_r64_delta']
    rows: list[list[str]] = []
    for task in tasks:
        g3_base = g3.get(('base', task))
        g3_v2   = g3.get(('v2',   task))
        g3_v3   = g3.get(('v3',   task))
        g4_base = g4.get(('base', task))
        g4_r32  = g4.get(('r32',  task))
        g4_r64  = g4.get(('r64',  task))

        def fmt(x):  return '' if x is None else f'{x:.4f}'
        def dlt(a, b): return '' if (a is None or b is None) else f'{(a-b):+.4f}'
        rows.append([
            task, KEY_METRICS.get(task, '?'),
            fmt(g3_base), fmt(g3_v2), dlt(g3_v2, g3_base),
            fmt(g3_v3),   dlt(g3_v3, g3_base),
            fmt(g4_base), fmt(g4_r32), dlt(g4_r32, g4_base),
            fmt(g4_r64),  dlt(g4_r64, g4_base),
        ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f'[paired] wrote {len(rows)} rows -> {args.out}')
    print('\n' + ' | '.join(header))
    for row in rows:
        print(' | '.join(row))


if __name__ == '__main__':
    main()
