"""Plot three-way comparison (base vs v2 vs v3) from the latest three-way snapshot.

Reads the most recent dir under D:/TRIBEV2/outputs/paper/eval_stats_three_way/
that contains base_outputs.json + v2_outputs.json + v3_outputs.json, scores each
sample via _score, and emits:

- three_way_style_transfer_bars.{png,svg} — grouped bar chart, 4 binary markers × 3 models
- three_way_continuous.{png,svg}           — grouped bars for word count and yeo7 alias
- three_way_per_sample.{png,svg}           — 3-panel heatmap, per-sample marker presence

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_three_way_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compute_eval_stats import _score  # type: ignore

SNAPSHOT_ROOT = Path('D:/TRIBEV2/outputs/paper/eval_stats_three_way')
OUT_DIR       = Path('D:/TRIBEV2/outputs/paper/figures')


def _latest_snapshot() -> Path | None:
    cands = []
    for p in SNAPSHOT_ROOT.iterdir() if SNAPSHOT_ROOT.exists() else []:
        if p.is_dir() and (p / 'base_outputs.json').exists() \
           and (p / 'v2_outputs.json').exists() and (p / 'v3_outputs.json').exists():
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def main() -> None:
    snap = _latest_snapshot()
    if snap is None:
        raise SystemExit(f'no three-way snapshot under {SNAPSHOT_ROOT}')
    base = json.loads((snap / 'base_outputs.json').read_text(encoding='utf-8'))
    v2   = json.loads((snap / 'v2_outputs.json').read_text(encoding='utf-8'))
    v3   = json.loads((snap / 'v3_outputs.json').read_text(encoding='utf-8'))
    n = min(len(base), len(v2), len(v3))
    print(f'[plot] snapshot={snap.name}  n={n}')

    base_s = [_score(r['completion']) for r in base[:n]]
    v2_s   = [_score(r['completion']) for r in v2[:n]]
    v3_s   = [_score(r['completion']) for r in v3[:n]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Panel 1: binary markers, grouped bars ---
    labels = ['opens "The stimulus"', 'TRIBE-v2 disclaimer', '"not a diagnostic"', 'peak time (seconds)']
    keys   = ['opens_with_template', 'has_tribe_disclaimer', 'has_not_diagnostic', 'mentions_peak_time']
    b  = [sum(float(s[k]) for s in base_s) / n for k in keys]
    v2r= [sum(float(s[k]) for s in v2_s)   / n for k in keys]
    v3r= [sum(float(s[k]) for s in v3_s)   / n for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, b,   w, label='base',            color='#9ecae1', edgecolor='#3182bd')
    ax.bar(x,     v2r, w, label='brain-v2 (r=32)', color='#fdae6b', edgecolor='#e6550d')
    ax.bar(x + w, v3r, w, label='brain-v3 (r=64)', color='#a1d99b', edgecolor='#31a354')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha='right')
    ax.set_ylim(0, 1.08); ax.set_ylabel('fraction of outputs')
    ax.set_title(f'Structural markers — held-out n={n}')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'three_way_style_transfer_bars.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'three_way_style_transfer_bars.svg', bbox_inches='tight')
    plt.close(fig)

    # --- Panel 2: continuous metrics (word count + yeo7 alias) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.5))
    def _bar3(ax, key, title):
        vals = [sum(float(s[key]) for s in rs) / n for rs in (base_s, v2_s, v3_s)]
        ax.bar([0,1,2], vals, color=['#9ecae1','#fdae6b','#a1d99b'],
               edgecolor=['#3182bd','#e6550d','#31a354'])
        ax.set_xticks([0,1,2]); ax.set_xticklabels(['base','v2','v3'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    _bar3(ax1, 'n_words', 'mean word count')
    _bar3(ax2, 'yeo7_any_alias', 'mean Yeo-7 alias mentions')
    _bar3(ax3, 'ttr', 'mean type-token ratio')
    fig.suptitle(f'Continuous metrics — held-out n={n}', y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'three_way_continuous.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'three_way_continuous.svg', bbox_inches='tight')
    plt.close(fig)

    # --- Panel 3: 3-panel per-sample heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    grids = [np.array([[int(s[k]) for k in keys] for s in rs]) for rs in (base_s, v2_s, v3_s)]
    for ax, grid, title in zip(axes, grids, ('base', 'brain-v2 (r=32)', 'brain-v3 (r=64)')):
        ax.imshow(grid, aspect='auto', cmap='Oranges', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_title(title)
    axes[0].set_ylabel(f'held-out prompt index (n={n})')
    fig.suptitle('Per-sample presence of target style markers', y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'three_way_per_sample.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'three_way_per_sample.svg', bbox_inches='tight')
    plt.close(fig)

    print(f'[plot] wrote three_way_* figures under {OUT_DIR}')


if __name__ == '__main__':
    main()
