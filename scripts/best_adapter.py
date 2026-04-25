"""Rank all trained adapters by benchmark performance vs their base model.

Answers "which adapter should I use?" with a ranked table that accounts for
stderr (runs at different limits aren't comparable at face value).

Produces two rankings:
  1. Raw mean Δ vs family base
  2. Significance-weighted: (# sig-positive tasks) - (# sig-negative tasks),
     tiebroken by mean Δ

Reads the same summary.csv files compile_bench_table.py does. Default output
goes to D:/research/BEST_ADAPTER.md.

Usage:
    python scripts/best_adapter.py
    python scripts/best_adapter.py --top-k 5
    python scripts/best_adapter.py --family gemma4
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from compile_bench_table import (  # noqa: E402
    _load_summaries, _group_by_variant, _delta_z, _z_marker,
)


def _family_prefix(slug: str) -> str:
    for fam in ('gemma4', 'gemma3', 'gemma2', 'llama3', 'qwen3', 'qwen2'):
        if slug.startswith(fam):
            return fam
    if slug.startswith('autoresearch'):
        return 'gemma4'
    return slug


def _score_entry(deltas: list[tuple[float | None, float | None]]) -> dict:
    """Compute raw mean Δ and significance-weighted score.

    sig_score = (# tasks with z >= +1) - (# tasks with z <= -1)
    A "clear winner" adapter has sig_score matching task count.
    """
    present = [d for d, _ in deltas if d is not None]
    mean_d = sum(present) / len(present) if present else 0.0
    sig_pos = sum(1 for d, z in deltas
                  if d is not None and z is not None and z >= 1.0)
    sig_neg = sum(1 for d, z in deltas
                  if d is not None and z is not None and z <= -1.0)
    strong_pos = sum(1 for d, z in deltas
                     if d is not None and z is not None and z >= 2.0)
    strong_neg = sum(1 for d, z in deltas
                     if d is not None and z is not None and z <= -2.0)
    return {
        'mean_d': mean_d,
        'sig_pos': sig_pos, 'sig_neg': sig_neg,
        'strong_pos': strong_pos, 'strong_neg': strong_neg,
        'sig_score': sig_pos - sig_neg,
        'n_tasks': sum(1 for d, _ in deltas if d is not None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-root', type=Path,
                    default=Path('D:/research/benchmarks'))
    ap.add_argument('--out', type=Path,
                    default=Path('D:/research/BEST_ADAPTER.md'))
    ap.add_argument('--family', type=str, default='',
                    help='restrict to a single family (e.g. gemma4)')
    ap.add_argument('--top-k', type=int, default=10)
    args = ap.parse_args()

    rows = _load_summaries(args.bench_root)
    if not rows:
        print(f'[best] no summaries in {args.bench_root}')
        return
    grouped = _group_by_variant(rows)
    tasks = sorted({r['task'] for r in rows})

    # Find a base per family (most recent by slug lex order == by ts).
    fam_to_base: dict[str, tuple[str, dict]] = {}
    for (slug, variant), vals in grouped.items():
        if variant != 'base':
            continue
        fam = _family_prefix(slug)
        if fam not in fam_to_base or slug > fam_to_base[fam][0]:
            fam_to_base[fam] = (slug, vals)

    # Score each non-base adapter vs its family base.
    candidates: list[dict] = []
    for (slug, variant), vals in grouped.items():
        if variant == 'base':
            continue
        fam = _family_prefix(slug)
        if args.family and fam != args.family:
            continue
        base = fam_to_base.get(fam)
        if base is None:
            continue
        base_slug, base_vals = base
        deltas: list[tuple[float | None, float | None]] = []
        for t in tasks:
            bp = base_vals.get(t)
            ap_ = vals.get(t)
            if bp is None or ap_ is None:
                deltas.append((None, None))
            else:
                bv, bse = bp
                av, ase = ap_
                deltas.append((av - bv, _delta_z(av, ase, bv, bse)))
        score = _score_entry(deltas)
        candidates.append({
            'slug': slug, 'variant': variant, 'family': fam,
            'base_slug': base_slug, 'deltas': deltas, **score,
        })

    if not candidates:
        print('[best] no candidates (need at least one base + one adapter per family)')
        return

    # Primary ranking: sig_score desc, mean_d desc, slug desc (newest first on tie)
    candidates.sort(key=lambda c: (c['sig_score'], c['mean_d'], c['slug']),
                    reverse=True)

    lines = [
        '# Best adapter ranking',
        '',
        f'_Compiled {time.strftime("%Y-%m-%d %H:%M:%S")}_  ',
        (f'_{len(candidates)} candidate(s) scored across {len(tasks)} tasks. '
         'Ranking is by significance score (# tasks where |z|≥1 and Δ>0 '
         'minus # tasks where |z|≥1 and Δ<0), tiebreak by mean Δ._'),
        '',
        '| rank | slug | variant | family | sig+ | sig− | strong+ | strong− | '
        'mean Δ | n_tasks |',
        '|------|------|---------|--------|------:|------:|------:|------:|------:|------:|',
    ]
    for i, c in enumerate(candidates[:args.top_k]):
        lines.append(
            f'| {i+1} | {c["slug"]} | {c["variant"]} | {c["family"]} | '
            f'{c["sig_pos"]} | {c["sig_neg"]} | '
            f'{c["strong_pos"]} | {c["strong_neg"]} | '
            f'**{c["mean_d"]:+.4f}** | {c["n_tasks"]} |'
        )

    lines.append('')
    lines.append('## Per-task breakdown (top candidate)')
    lines.append('')
    if candidates:
        top = candidates[0]
        lines.append(f'**{top["slug"]} / {top["variant"]}** '
                     f'vs `{top["base_slug"]}`:')
        lines.append('')
        lines.append('| task | Δ | z | sig |')
        lines.append('|------|------:|------:|:---|')
        for t, (d, z) in zip(tasks, top['deltas']):
            if d is None:
                lines.append(f'| {t} | — | — | — |')
            else:
                mark = _z_marker(z)
                zstr = f'{z:+.2f}' if z is not None else '—'
                lines.append(f'| {t} | {d:+.4f} | {zstr} | {mark or "·"} |')

    lines.append('')
    lines.append('## Recommendation')
    lines.append('')
    top = candidates[0]
    if top['sig_pos'] >= 3 and top['sig_neg'] == 0:
        verdict = ('**ESCALATE**: clear winner — multiple tasks at |z|≥1 with '
                   'no significant regressions. Safe to 31B full train.')
    elif top['sig_pos'] >= 1 and top['sig_pos'] > top['sig_neg']:
        verdict = ('**CONDITIONAL**: partial win. Consider a longer-limit '
                   'rerun (limit=500+) to tighten bounds before 31B.')
    elif top['sig_pos'] == 0 and top['sig_neg'] == 0:
        verdict = ('**INCONCLUSIVE**: all deltas are within noise. Need more '
                   'samples (raise bench --limit) or a different axis.')
    else:
        verdict = ('**REJECT**: top candidate has net-negative significant '
                   'deltas. Do not escalate; try a different hypothesis.')
    lines.append(verdict)

    args.out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[best] wrote {args.out}')
    print('')
    print(f'[best] top: {top["slug"]} / {top["variant"]} — '
          f'mean Δ {top["mean_d"]:+.4f}, sig+/− {top["sig_pos"]}/{top["sig_neg"]}')
    print(f'[best]   {verdict.replace(chr(10), " ").replace("**", "")}')


if __name__ == '__main__':
    main()
