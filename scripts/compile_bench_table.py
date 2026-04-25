"""Compile a single paired benchmark table across all our trained adapters.

Reads every lm-eval summary.csv under D:/research/benchmarks/ and produces:

  - D:/research/BENCH_MATRIX.md      (markdown table)
  - D:/research/BENCH_MATRIX.csv     (flat csv for downstream plotting)
  - D:/research/BENCH_DELTAS.md      (deltas vs each base model)

Usage:
    python scripts/compile_bench_table.py
    python scripts/compile_bench_table.py --out-root D:/research
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


METRIC_PREFS = {
    'arc_challenge': ['acc_norm,none', 'acc,none'],
    'arc_easy': ['acc_norm,none', 'acc,none'],
    'piqa': ['acc_norm,none', 'acc,none'],
    'openbookqa': ['acc_norm,none', 'acc,none'],
    'gsm8k': ['exact_match,strict-match', 'exact_match,flexible-extract',
              'acc,none'],
    'truthfulqa_mc1': ['acc,none', 'acc_norm,none'],
    'hellaswag': ['acc_norm,none', 'acc,none'],
    'winogrande': ['acc,none'],
    'mmlu': ['acc,none'],
}


def _stderr_key_for(metric_key: str) -> str:
    """Map 'acc,none' -> 'acc_stderr,none', 'exact_match,strict-match' -> 'exact_match_stderr,strict-match'."""
    name, _, suffix = metric_key.partition(',')
    return f'{name}_stderr,{suffix}' if suffix else f'{name}_stderr'


def _as_float(v) -> float | None:
    if v is None or v == '' or v == 'None':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_metric(row: dict) -> tuple[str, float, float | None] | None:
    task = row.get('task', '')
    for key in METRIC_PREFS.get(task, ['acc_norm,none', 'acc,none',
                                       'exact_match,strict-match',
                                       'exact_match,flexible-extract']):
        v = _as_float(row.get(key))
        if v is None:
            continue
        se = _as_float(row.get(_stderr_key_for(key)))
        return key, v, se
    return None


def _pick_from_metric_dict(task: str,
                           metrics: dict) -> tuple[str, float, float | None] | None:
    for key in METRIC_PREFS.get(task, ['acc_norm,none', 'acc,none',
                                       'exact_match,strict-match',
                                       'exact_match,flexible-extract']):
        v = _as_float(metrics.get(key))
        if v is None:
            continue
        se = _as_float(metrics.get(_stderr_key_for(key)))
        return key, v, se
    return None


def _load_summaries(bench_root: Path) -> list[dict]:
    """Each returned dict: {slug, variant, task, metric, value, stderr, path}."""
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    # Prefer summary.csv (it's the authoritative post-hoc aggregation)
    for summary in bench_root.glob('*/summary.csv'):
        slug = summary.parent.name
        try:
            with summary.open(newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    task = row.get('task', '')
                    if not task:
                        continue
                    picked = _pick_metric(row)
                    if not picked:
                        continue
                    metric, val, se = picked
                    variant = row.get('variant', '?')
                    seen.add((slug, variant, task))
                    out.append({
                        'slug': slug,
                        'variant': variant,
                        'task': task,
                        'metric': metric,
                        'value': val,
                        'stderr': se,
                        'n': row.get('n_samples', ''),
                        'path': str(summary),
                    })
        except Exception as e:
            print(f'[bench] WARN: failed to parse {summary}: {e}')

    # Fallback: per-variant JSON results (lm-eval writes results_<ts>.json)
    for results_json in bench_root.glob('*/*/**/results_*.json'):
        parts = results_json.relative_to(bench_root).parts
        if len(parts) < 3:
            continue
        slug = parts[0]
        variant = parts[1]
        try:
            d = json.loads(results_json.read_text(encoding='utf-8'))
        except Exception as e:
            print(f'[bench] WARN: failed to parse {results_json}: {e}')
            continue
        results = d.get('results') or {}
        for task, metrics in results.items():
            if (slug, variant, task) in seen:
                continue
            picked = _pick_from_metric_dict(task, metrics)
            if not picked:
                continue
            metric, val, se = picked
            seen.add((slug, variant, task))
            out.append({
                'slug': slug,
                'variant': variant,
                'task': task,
                'metric': metric,
                'value': val,
                'stderr': se,
                'n': str((d.get('n-samples', {}) or {}).get(task, '')),
                'path': str(results_json),
            })
    return out


def _group_by_variant(rows: list[dict]) -> dict[tuple[str, str],
                                                   dict[str, tuple[float, float | None]]]:
    """Returns {(slug, variant): {task: (value, stderr)}}."""
    out: dict[tuple[str, str], dict[str, tuple[float, float | None]]] = defaultdict(dict)
    for r in rows:
        out[(r['slug'], r['variant'])][r['task']] = (r['value'], r.get('stderr'))
    return out


def _z_marker(z: float | None) -> str:
    """Return a compact sig marker: '**' |z|>=2, '*' |z|>=1, '' otherwise."""
    if z is None:
        return ''
    az = abs(z)
    if az >= 2.0:
        return '**'
    if az >= 1.0:
        return '*'
    return ''


def _delta_z(av: float, av_se: float | None,
             bv: float, bv_se: float | None) -> float | None:
    """Z-score for av - bv, treating the two benchmark runs as independent.
    Returns None if either stderr is missing or combined stderr is 0."""
    if av_se is None or bv_se is None:
        return None
    combined = math.sqrt(av_se * av_se + bv_se * bv_se)
    if combined <= 0:
        return None
    return (av - bv) / combined


def _identify_base_variant(grouped: dict) -> tuple[str, str] | None:
    """The first (slug, 'base') key we find."""
    for key in grouped:
        if key[1] == 'base':
            return key
    return None


def _fmt_pct(v: float) -> str:
    return f'{v*100:.2f}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-root', type=Path,
                    default=Path('D:/research/benchmarks'))
    ap.add_argument('--out-root', type=Path, default=Path('D:/research'))
    args = ap.parse_args()

    rows = _load_summaries(args.bench_root)
    if not rows:
        print(f'[bench] no summaries in {args.bench_root}')
        return

    grouped = _group_by_variant(rows)
    tasks = sorted({r['task'] for r in rows})
    keys = sorted(grouped.keys())

    # Flat CSV — one row per (slug, variant), columns: task value + task stderr
    csv_out = args.out_root / 'BENCH_MATRIX.csv'
    with csv_out.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['slug', 'variant']
        for t in tasks:
            header.extend([t, f'{t}_se'])
        header.append('mean')
        w.writerow(header)
        for key in keys:
            vals = grouped[key]
            row = [key[0], key[1]]
            present = []
            for t in tasks:
                pair = vals.get(t)
                if pair is None:
                    row.extend(['', ''])
                else:
                    v, se = pair
                    row.extend([f'{v:.4f}',
                                f'{se:.4f}' if se is not None else ''])
                    present.append(v)
            mean = sum(present) / len(present) if present else 0
            row.append(f'{mean:.4f}')
            w.writerow(row)
    print(f'[bench] wrote {csv_out}')

    # Matrix markdown
    lines = [
        '# Benchmark matrix',
        '',
        f'_Compiled {time.strftime("%Y-%m-%d %H:%M:%S")}_  ',
        f'_{len(keys)} (slug, variant) pairs across {len(tasks)} tasks_  ',
        '_Values shown as `acc ± stderr`; `—` means no data._',
        '',
        '| slug | variant | ' + ' | '.join(tasks) + ' | mean |',
        '|------|---------|' + '|'.join(['------:'] * (len(tasks) + 1)) + '|',
    ]
    # Sort by mean desc
    key_means: list[tuple[tuple[str, str], float]] = []
    for k in keys:
        vals = grouped[k]
        present = [pair[0] for pair in vals.values() if pair is not None]
        m = sum(present) / len(present) if present else 0
        key_means.append((k, m))
    key_means.sort(key=lambda kv: kv[1], reverse=True)
    for (slug, variant), mean in key_means:
        vals = grouped[(slug, variant)]
        xs = []
        for t in tasks:
            pair = vals.get(t)
            if pair is None:
                xs.append('—')
            else:
                v, se = pair
                if se is not None:
                    xs.append(f'{v:.3f}±{se:.3f}')
                else:
                    xs.append(f'{v:.4f}')
        lines.append(f'| {slug} | {variant} | ' + ' | '.join(xs) +
                     f' | **{mean:.4f}** |')

    (args.out_root / 'BENCH_MATRIX.md').write_text('\n'.join(lines),
                                                    encoding='utf-8')
    print(f'[bench] wrote {args.out_root / "BENCH_MATRIX.md"}')

    def _family_prefix(slug: str) -> str:
        """gemma3-* -> gemma3, gemma4-* -> gemma4, autoresearch-* -> gemma4
        (since the autoresearch loop defaults to gemma-4 e4b base), otherwise
        the whole slug."""
        for fam in ('gemma4', 'gemma3', 'gemma2', 'llama3', 'qwen3', 'qwen2'):
            if slug.startswith(fam):
                return fam
        if slug.startswith('autoresearch'):
            return 'gemma4'
        return slug

    # Per-family deltas: keep most recent 'base' per family; compute
    # (adapter - base) and z-score per task.
    fam_to_base: dict[str, tuple[str, dict[str, tuple[float, float | None]]]] = {}
    for (slug, variant), vals in grouped.items():
        if variant != 'base':
            continue
        fam = _family_prefix(slug)
        prev = fam_to_base.get(fam)
        if prev is None or slug > prev[0]:
            fam_to_base[fam] = (slug, vals)

    delta_lines = [
        '# Benchmark deltas (adapter − base, grouped by model family)',
        '',
        f'_Compiled {time.strftime("%Y-%m-%d %H:%M:%S")}_  ',
        ('_Adapter values come from any slug in the family; base is the '
         'most recent limit-matched base run in that family._  '),
        ('_Cell format: `+/-delta` with `*` when |z|≥1 and `**` when |z|≥2 '
         '(adapter vs base, independent stderr)._'),
        '',
    ]
    for fam, (base_slug, base_vals) in sorted(fam_to_base.items()):
        delta_lines.append(f'## {fam} family (base: {base_slug})')
        delta_lines.append('')
        delta_lines.append('| slug | variant | '
                           + ' | '.join(tasks) + ' | mean Δ | sig tasks |')
        delta_lines.append('|------|---------|'
                           + '|'.join(['------:'] * (len(tasks) + 2)) + '|')
        rows = [(s, v, grouped[(s, v)])
                for (s, v) in sorted(grouped.keys())
                if _family_prefix(s) == fam and v != 'base']
        # Sort each family by mean Δ so the strongest adapter bubbles up
        scored_rows = []
        for slug, variant, vals in rows:
            deltas: list[tuple[float | None, float | None]] = []
            for t in tasks:
                bpair = base_vals.get(t)
                apair = vals.get(t)
                if bpair is None or apair is None:
                    deltas.append((None, None))
                else:
                    bv, bse = bpair
                    av, ase = apair
                    deltas.append((av - bv, _delta_z(av, ase, bv, bse)))
            present = [d for d, _ in deltas if d is not None]
            mean_d = sum(present) / len(present) if present else 0
            scored_rows.append((slug, variant, deltas, mean_d))
        scored_rows.sort(key=lambda r: r[3], reverse=True)
        for slug, variant, deltas, mean_d in scored_rows:
            cells = []
            sig_hits = 0
            for d, z in deltas:
                if d is None:
                    cells.append('—')
                else:
                    mark = _z_marker(z)
                    if z is not None and abs(z) >= 1.0:
                        sig_hits += 1
                    cells.append(f'{d:+.3f}{mark}')
            delta_lines.append(f'| {slug} | {variant} | '
                               + ' | '.join(cells)
                               + f' | **{mean_d:+.4f}** | {sig_hits}/{len(tasks)} |')
        delta_lines.append('')

    (args.out_root / 'BENCH_DELTAS.md').write_text('\n'.join(delta_lines),
                                                    encoding='utf-8')
    print(f'[bench] wrote {args.out_root / "BENCH_DELTAS.md"}')


if __name__ == '__main__':
    main()
