"""Summarize autoresearch runs into a human-readable report.

Reads `D:/research/autoresearch/results/*.json` and produces:
  - LEADERBOARD.md (already written by autoresearch_loop.py after each iter;
    this script is a more detailed post-hoc view)
  - REPORT.md with per-axis analysis (LR effect, rank effect, mix effect)
  - SCORECARD.csv for downstream plotting

Usage:
    python scripts/autoresearch_report.py
    python scripts/autoresearch_report.py --root D:/research/autoresearch

Safe to run anytime, even while the loop is still running.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _score(h: dict) -> float:
    acc_keys = [k for k in h if k.startswith('bench_') and k.endswith('_acc_norm')]
    acc_keys += [k for k in h if k.startswith('bench_') and k.endswith('_acc')
                 and k.replace('_acc', '_acc_norm') not in h]
    em_keys = [k for k in h if k.startswith('bench_') and k.endswith('_exact_match')]
    bench_vals = [h[k] for k in acc_keys + em_keys if isinstance(h.get(k), (int, float))]
    bench_mean = sum(bench_vals) / max(1, len(bench_vals)) if bench_vals else 0.0
    spread = max(0.0, h.get('tier_fk_spread', 0.0))
    return bench_mean + min(0.05, spread * 0.01)


def _bench_mean(h: dict) -> float | None:
    keys = ([k for k in h if k.startswith('bench_') and k.endswith('_acc_norm')]
            + [k for k in h if k.startswith('bench_') and k.endswith('_acc')
               and k.replace('_acc', '_acc_norm') not in h]
            + [k for k in h if k.startswith('bench_') and k.endswith('_exact_match')])
    vals = [h[k] for k in keys if isinstance(h.get(k), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_all(root: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted((root / 'results').glob('*.json')):
        try:
            out.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception as e:
            print(f'[report] WARN: could not parse {p}: {e}')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('D:/research/autoresearch'))
    args = ap.parse_args()

    history = _load_all(args.root)
    if not history:
        print(f'[report] no results in {args.root / "results"}')
        return

    history.sort(key=_score, reverse=True)

    # CSV scorecard
    scorecard = args.root / 'SCORECARD.csv'
    fieldnames = ['rank', 'id', 'name', 'score', 'bench_mean',
                  'lora_r', 'lora_alpha', 'lr', 'smoke_steps',
                  'weights', 'fk_spread', 'student_fk', 'public_fk', 'expert_fk',
                  'student_overlap', 'public_overlap', 'expert_overlap',
                  'elapsed_min', 'phase_failed']
    with scorecard.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, h in enumerate(history):
            hyp = h.get('hypothesis', {})
            tr = hyp.get('training') or {}
            cu = hyp.get('curriculum') or {}
            w.writerow({
                'rank': i + 1,
                'id': hyp.get('id', ''),
                'name': hyp.get('name', ''),
                'score': f'{_score(h):.4f}',
                'bench_mean': f'{_bench_mean(h):.4f}' if _bench_mean(h) is not None else '',
                'lora_r': tr.get('lora-r', ''),
                'lora_alpha': tr.get('lora-alpha', ''),
                'lr': tr.get('lr', ''),
                'smoke_steps': tr.get('smoke-steps', ''),
                'weights': cu.get('weights', ''),
                'fk_spread': f'{h.get("tier_fk_spread", 0):+.3f}' if 'tier_fk_spread' in h else '',
                'student_fk': f'{h.get("tier_student_fk_median", 0):.2f}' if 'tier_student_fk_median' in h else '',
                'public_fk':  f'{h.get("tier_public_fk_median", 0):.2f}'  if 'tier_public_fk_median'  in h else '',
                'expert_fk':  f'{h.get("tier_expert_fk_median", 0):.2f}'  if 'tier_expert_fk_median'  in h else '',
                'student_overlap': f'{h.get("tier_student_overlap_median", 0):.3f}' if 'tier_student_overlap_median' in h else '',
                'public_overlap':  f'{h.get("tier_public_overlap_median", 0):.3f}'  if 'tier_public_overlap_median'  in h else '',
                'expert_overlap':  f'{h.get("tier_expert_overlap_median", 0):.3f}'  if 'tier_expert_overlap_median'  in h else '',
                'elapsed_min': f'{h.get("elapsed_s", 0) / 60:.1f}',
                'phase_failed': h.get('phase_failed', ''),
            })

    # Markdown report
    lines = [
        '# Autoresearch report',
        '',
        f'_Generated {time.strftime("%Y-%m-%d %H:%M:%S")}_  ',
        f'_Total experiments: **{len(history)}**  ·  '
        f'Completed: **{sum(1 for h in history if not h.get("phase_failed"))}**_',
        '',
        '## Top 10',
        '',
        '| rank | id | name | score | bench | FK spread | min |',
        '|------|----|------|------:|------:|----------:|----:|',
    ]
    for i, h in enumerate(history[:10]):
        hyp = h.get('hypothesis', {})
        bm = _bench_mean(h)
        lines.append(
            f'| {i+1} | {hyp.get("id", "")} | {hyp.get("name", "")} | '
            f'{_score(h):.4f} | {bm:.4f if bm is not None else ""} | '
            f'{h.get("tier_fk_spread", 0):+.2f} | '
            f'{h.get("elapsed_s", 0)/60:.1f} |'
        )
    # Axis analyses
    axes = defaultdict(list)
    for h in history:
        if h.get('phase_failed'):
            continue
        hyp = h.get('hypothesis', {})
        tr = hyp.get('training') or {}
        cu = hyp.get('curriculum') or {}
        s = _score(h)
        if 'lora-r' in tr:
            axes[('rank', str(tr['lora-r']))].append(s)
        if 'lr' in tr:
            axes[('lr', f'{float(tr["lr"]):.0e}')].append(s)
        if 'smoke-steps' in tr:
            axes[('steps', str(tr['smoke-steps']))].append(s)
        if 'weights' in cu:
            axes[('weights', str(cu['weights']))].append(s)

    lines.append('')
    lines.append('## Per-axis mean score (higher = better)')
    for axis in ('rank', 'lr', 'steps', 'weights'):
        vals = [(k, v) for (a, k), v in axes.items() if a == axis]
        if not vals:
            continue
        vals.sort(key=lambda x: statistics.mean(x[1]), reverse=True)
        lines.append('')
        lines.append(f'### {axis}')
        lines.append('')
        lines.append('| value | n | mean score | std |')
        lines.append('|-------|--:|-----------:|----:|')
        for k, v in vals:
            m = statistics.mean(v)
            sd = statistics.pstdev(v) if len(v) > 1 else 0
            lines.append(f'| `{k}` | {len(v)} | {m:.4f} | {sd:.4f} |')

    # Failures
    failures = [h for h in history if h.get('phase_failed')]
    if failures:
        lines.append('')
        lines.append(f'## Failures ({len(failures)})')
        lines.append('')
        lines.append('| id | phase | notes |')
        lines.append('|----|-------|-------|')
        for h in failures:
            hyp = h.get('hypothesis', {})
            lines.append(f'| {hyp.get("id", "?")} | {h.get("phase_failed")} | '
                         f'{h.get("bench_rc", "")} {h.get("tier_rc", "")} |')

    (args.root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[report] wrote {args.root / "REPORT.md"}')
    print(f'[report] wrote {scorecard}')

    # Quick stdout preview
    print('')
    print('=== Top 5 ===')
    for i, h in enumerate(history[:5]):
        hyp = h.get('hypothesis', {})
        print(f'  {i+1}. {hyp.get("id", ""):30s}  '
              f'score={_score(h):.4f}  bench={_bench_mean(h) or 0:.4f}  '
              f'fk_spread={h.get("tier_fk_spread", 0):+.2f}')


if __name__ == '__main__':
    main()
