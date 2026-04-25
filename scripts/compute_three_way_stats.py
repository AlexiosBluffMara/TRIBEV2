"""Three-way eval stats: base vs v2-adapter vs v3-adapter.

Expects a snapshot dir with:
    picks.json  base_outputs.json  v2_outputs.json  v3_outputs.json

Reuses _score / _bootstrap_ci / _paired_sign_test from compute_eval_stats.py.

Emits under D:/TRIBEV2/outputs/paper/eval_stats_three_way/:
    eval_stats_three_way.md
    eval_stats_three_way.json
    eval_stats_three_way.csv

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/compute_three_way_stats.py \\
        --snap D:/TRIBEV2/outputs/paper/eval_stats_three_way/eval_three_way_<ts>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent))
from compute_eval_stats import _score, _bootstrap_ci, _paired_sign_test  # type: ignore

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = Path('D:/TRIBEV2/outputs/paper/eval_stats_three_way')

KEYS_BIN = ['opens_with_template', 'has_tribe_disclaimer', 'has_not_diagnostic', 'mentions_peak_time']
KEYS_NUM = ['yeo7_networks_mentioned', 'yeo7_any_alias', 'roi_verbatim_count', 'n_words', 'n_chars', 'ttr']


def _load(snap: Path) -> tuple[list[dict], list[dict], list[dict]]:
    base = json.loads((snap / 'base_outputs.json').read_text(encoding='utf-8'))
    v2   = json.loads((snap / 'v2_outputs.json').read_text(encoding='utf-8'))
    v3   = json.loads((snap / 'v3_outputs.json').read_text(encoding='utf-8'))
    return base, v2, v3


def _summary(rows_scored: list[dict]) -> dict:
    out: dict = {}
    for k in KEYS_BIN:
        out[k + '_rate'] = mean(float(r[k]) for r in rows_scored)
    for k in KEYS_NUM:
        vals = [float(r[k]) for r in rows_scored]
        out[k + '_mean'] = mean(vals) if vals else 0.0
        out[k + '_std']  = stdev(vals) if len(vals) >= 2 else 0.0
    return out


def _paired_block(a: list[dict], b: list[dict]) -> dict:
    """Paired deltas (a - b) across matching samples; returns mean/std/CI/sign-test per metric."""
    deltas: dict[str, list[float]] = {}
    for ai, bi in zip(a, b):
        deltas.setdefault('opens_with_template_delta', []).append(int(ai['opens_with_template']) - int(bi['opens_with_template']))
        deltas.setdefault('has_tribe_disclaimer_delta', []).append(int(ai['has_tribe_disclaimer']) - int(bi['has_tribe_disclaimer']))
        deltas.setdefault('has_not_diagnostic_delta', []).append(int(ai['has_not_diagnostic']) - int(bi['has_not_diagnostic']))
        deltas.setdefault('mentions_peak_time_delta', []).append(int(ai['mentions_peak_time']) - int(bi['mentions_peak_time']))
        deltas.setdefault('yeo7_strict_delta', []).append(ai['yeo7_networks_mentioned'] - bi['yeo7_networks_mentioned'])
        deltas.setdefault('yeo7_alias_delta', []).append(ai['yeo7_any_alias'] - bi['yeo7_any_alias'])
        deltas.setdefault('roi_verbatim_delta', []).append(ai['roi_verbatim_count'] - bi['roi_verbatim_count'])
        deltas.setdefault('n_words_delta', []).append(ai['n_words'] - bi['n_words'])
        deltas.setdefault('ttr_delta', []).append(ai['ttr'] - bi['ttr'])
    out: dict = {}
    for k, vs in deltas.items():
        out[k] = {
            'mean': mean(vs),
            'std':  stdev(vs) if len(vs) >= 2 else 0.0,
            'ci95': list(_bootstrap_ci(vs)),
            'sign': _paired_sign_test(vs),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--snap', type=Path, required=True,
                    help='Snapshot dir with base_outputs/v2_outputs/v3_outputs .json')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_rows, v2_rows, v3_rows = _load(args.snap)

    # align by prompt
    n = min(len(base_rows), len(v2_rows), len(v3_rows))
    base_scored = [_score(r['completion']) for r in base_rows[:n]]
    v2_scored   = [_score(r['completion']) for r in v2_rows[:n]]
    v3_scored   = [_score(r['completion']) for r in v3_rows[:n]]

    base_sum = _summary(base_scored)
    v2_sum   = _summary(v2_scored)
    v3_sum   = _summary(v3_scored)

    v2_v_base = _paired_block(v2_scored, base_scored)
    v3_v_base = _paired_block(v3_scored, base_scored)
    v3_v_v2   = _paired_block(v3_scored, v2_scored)

    lines: list[str] = []
    lines.append(f'# Three-way eval: base vs v2 vs v3 (n={n} paired samples)\n')
    lines.append(f'Snapshot: `{args.snap}`\n')
    lines.append('## Style-transfer rates (binary)\n')
    lines.append('| metric | base | v2 | v3 | Δ(v2-base) | Δ(v3-base) | Δ(v3-v2) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for k in KEYS_BIN:
        r_b = base_sum[k + '_rate']; r_v2 = v2_sum[k + '_rate']; r_v3 = v3_sum[k + '_rate']
        lines.append(f'| {k} | {r_b:.3f} | {r_v2:.3f} | {r_v3:.3f} | '
                     f'{r_v2-r_b:+.3f} | {r_v3-r_b:+.3f} | {r_v3-r_v2:+.3f} |')
    lines.append('\n## Continuous metrics (mean ± std)\n')
    lines.append('| metric | base | v2 | v3 |')
    lines.append('|---|---:|---:|---:|')
    for k in KEYS_NUM:
        lines.append(f'| {k} | {base_sum[k+"_mean"]:.3f} ± {base_sum[k+"_std"]:.3f} | '
                     f'{v2_sum[k+"_mean"]:.3f} ± {v2_sum[k+"_std"]:.3f} | '
                     f'{v3_sum[k+"_mean"]:.3f} ± {v3_sum[k+"_std"]:.3f} |')

    for label, block in (('v2 vs base', v2_v_base), ('v3 vs base', v3_v_base), ('v3 vs v2', v3_v_v2)):
        lines.append(f'\n## Paired deltas — {label} (95% bootstrap CI + two-sided sign test)\n')
        lines.append('| metric | mean ± std | 95% CI | sign-test p |')
        lines.append('|---|---:|---:|---:|')
        for k, v in block.items():
            lo, hi = v['ci95']
            p = v['sign']['p_two_sided']
            p_str = f'{p:.3g}' if isinstance(p, float) else str(p)
            lines.append(f"| {k} | {v['mean']:+.3f} ± {v['std']:.3f} | [{lo:+.3f}, {hi:+.3f}] | {p_str} |")

    md = '\n'.join(lines) + '\n'
    out_md   = OUT_DIR / 'eval_stats_three_way.md'
    out_json = OUT_DIR / 'eval_stats_three_way.json'
    out_md.write_text(md, encoding='utf-8')

    combined = {
        'n_pairs': n,
        'snapshot': str(args.snap),
        'base_summary': base_sum,
        'v2_summary':   v2_sum,
        'v3_summary':   v3_sum,
        'v2_vs_base':   v2_v_base,
        'v3_vs_base':   v3_v_base,
        'v3_vs_v2':     v3_v_v2,
    }
    out_json.write_text(json.dumps(combined, indent=2, default=lambda o: o if not hasattr(o, '__dict__') else o.__dict__), encoding='utf-8')

    print(md)
    print(f'[stats] wrote {out_md}')
    print(f'[stats] wrote {out_json}')


if __name__ == '__main__':
    main()
