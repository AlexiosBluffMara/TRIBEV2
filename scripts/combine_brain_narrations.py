"""Combine and dedupe all brain_narrations_*.jsonl files into a single corpus.

Dedupes on full prompt text. Emits brain_narrations_combined_<total>.jsonl into
D:/research/datasets/.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/combine_brain_narrations.py
"""
from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path('D:/research/datasets')


def main() -> None:
    inputs = sorted(DATA_DIR.glob('brain_narrations_*.jsonl'))
    inputs = [p for p in inputs if 'combined' not in p.name]
    print(f'[combine] inputs = {[p.name for p in inputs]}')

    seen: dict[str, dict] = {}
    per_file: dict[str, int] = {}
    for p in inputs:
        rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        added = 0
        for r in rows:
            k = r['prompt']
            if k in seen:
                continue
            seen[k] = r
            added += 1
        per_file[p.name] = added
        print(f'  {p.name}: +{added}/{len(rows)}')

    rows_out = list(seen.values())
    n = len(rows_out)
    out = DATA_DIR / f'brain_narrations_combined_{n}.jsonl'
    with out.open('w', encoding='utf-8') as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[combine] wrote {out}  unique_rows={n}')


if __name__ == '__main__':
    main()
