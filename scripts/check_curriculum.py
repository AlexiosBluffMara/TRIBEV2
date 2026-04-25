"""Sanity-check a curriculum jsonl before spending GPU time on it.

Catches the kinds of things that silently destroy a training run:
  - empty prompts or completions
  - pathologically long completions (>10k chars → likely corruption)
  - bad tier / signal labels
  - duplicate (prompt, completion) pairs
  - a single source dominating >60% of rows (unless that's the intent)
  - near-duplicate completions (same 200-char prefix repeated >5x)

Exits non-zero if any FATAL check fails, so it's safe to use as a gate
in shell scripts:
    python scripts/check_curriculum.py path.jsonl || exit 1

Usage:
    python scripts/check_curriculum.py <path>
    python scripts/check_curriculum.py <path> --max-completion-chars 8000
    python scripts/check_curriculum.py <path> --samples-per-source 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


VALID_SIGNALS = {'A', 'B', 'C', 'D'}
VALID_TIERS = {'student', 'public', 'expert'}


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise SystemExit(f'[check] FATAL: line {i} unparseable: {e}')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=Path)
    ap.add_argument('--max-completion-chars', type=int, default=10000,
                    help='completions above this length are flagged')
    ap.add_argument('--min-prompt-chars', type=int, default=20)
    ap.add_argument('--samples-per-source', type=int, default=1)
    ap.add_argument('--max-source-share', type=float, default=0.60,
                    help='fatal if any single source exceeds this fraction')
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if not args.path.exists():
        raise SystemExit(f'[check] FATAL: {args.path} does not exist')

    rows = _load(args.path)
    print(f'[check] loaded {len(rows)} rows from {args.path.name}')

    errors: list[str] = []
    warnings: list[str] = []

    # --- Schema + length checks ---
    empty_prompt = 0
    empty_completion = 0
    short_prompt = 0
    huge_completion = 0
    bad_signal = 0
    bad_tier = 0
    for i, r in enumerate(rows):
        prompt = r.get('prompt') or ''
        completion = r.get('completion') or ''
        if not prompt:
            empty_prompt += 1
        elif len(prompt) < args.min_prompt_chars:
            short_prompt += 1
        if not completion:
            empty_completion += 1
        elif len(completion) > args.max_completion_chars:
            huge_completion += 1
        if r.get('signal') not in VALID_SIGNALS:
            bad_signal += 1
        if r.get('tier') not in VALID_TIERS:
            bad_tier += 1

    if empty_prompt:
        errors.append(f'empty prompts: {empty_prompt}')
    if empty_completion:
        errors.append(f'empty completions: {empty_completion}')
    if bad_signal:
        errors.append(f'rows with bad signal: {bad_signal}')
    if bad_tier:
        errors.append(f'rows with bad tier: {bad_tier}')
    if short_prompt:
        warnings.append(f'short prompts (<{args.min_prompt_chars} chars): '
                        f'{short_prompt}')
    if huge_completion:
        warnings.append(f'huge completions (>{args.max_completion_chars} chars): '
                        f'{huge_completion}')

    # --- Distribution ---
    signals = Counter(r.get('signal') for r in rows)
    tiers = Counter(r.get('tier') for r in rows)
    sources = Counter(r.get('source') for r in rows)
    print(f'[check] signal: {dict(signals)}')
    print(f'[check] tier:   {dict(tiers)}')
    print(f'[check] sources ({len(sources)}):')
    for src, n in sources.most_common():
        share = n / len(rows)
        marker = ' ⚠' if share > args.max_source_share else ''
        print(f'    {src:32s}  {n:6d}  ({share:5.1%}){marker}')
        if share > args.max_source_share:
            errors.append(f'source {src!r} dominates: {share:.1%} > '
                          f'{args.max_source_share:.0%}')

    # --- Exact-duplicate (prompt, completion) pairs ---
    seen_pairs: set[tuple[str, str]] = set()
    exact_dups = 0
    for r in rows:
        key = (r.get('prompt', ''), r.get('completion', ''))
        if key in seen_pairs:
            exact_dups += 1
        else:
            seen_pairs.add(key)
    if exact_dups:
        warnings.append(f'exact-duplicate (prompt, completion) pairs: {exact_dups}')

    # --- Near-duplicate completions (same first 200 chars repeating heavily) ---
    completion_prefix = Counter()
    for r in rows:
        c = (r.get('completion') or '')[:200]
        if c:
            completion_prefix[c] += 1
    top_prefix, top_count = completion_prefix.most_common(1)[0] if completion_prefix else ('', 0)
    if top_count > 5:
        warnings.append(f'most-common completion prefix appears {top_count}x: '
                        f'{top_prefix[:80]!r}')

    # --- Completion length histogram ---
    lens = sorted(len(r.get('completion') or '') for r in rows)
    p50 = lens[len(lens) // 2]
    p90 = lens[int(len(lens) * 0.9)]
    p99 = lens[int(len(lens) * 0.99)]
    print(f'[check] completion chars: p50={p50} p90={p90} p99={p99} max={lens[-1]}')

    # --- Sample per source ---
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r.get('source', ''), []).append(r)
    if args.samples_per_source > 0:
        print(f'[check] {args.samples_per_source} sample(s) per source:')
        for src in sorted(by_source):
            for r in by_source[src][:args.samples_per_source]:
                p = r.get('prompt', '').replace('\n', ' ')[:110]
                c = r.get('completion', '').replace('\n', ' ')[:110]
                print(f'  [{src}] {r.get("signal")}/{r.get("tier")}')
                print(f'    P: {p}')
                print(f'    C: {c}')

    # --- Summary ---
    print('')
    if warnings:
        print(f'[check] WARNINGS ({len(warnings)}):')
        for w in warnings:
            print(f'  ⚠ {w}')
    if errors:
        print(f'[check] ERRORS ({len(errors)}):')
        for e in errors:
            print(f'  ✗ {e}')
        print('\n[check] FATAL — refusing to proceed')
        sys.exit(1)
    print(f'[check] OK  ({len(rows)} rows, {len(sources)} sources, '
          f'{len(warnings)} warnings, 0 errors)')


if __name__ == '__main__':
    main()
