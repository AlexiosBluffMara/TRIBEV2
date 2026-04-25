"""Plot n=30 expanded-eval style-transfer comparison (base vs brain-v2 adapter).

Reads from the expanded-eval snapshot dir (the most recent under
D:/TRIBEV2/outputs/paper/eval_stats_n30/eval_brain_llamacpp_*), computes
per-sample scores with the same regex set used by compute_eval_stats.py, and
emits:

- n30_style_transfer_bars.{png,svg} — base vs adapter, structural markers
- n30_per_sample.{png,svg}           — per-prompt binary-metric heatmap

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_n30_comparison.py
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

SNAPSHOT_ROOT = Path('D:/TRIBEV2/outputs/paper/eval_stats_n30')
OUT_DIR       = Path('D:/TRIBEV2/outputs/paper/figures')


def _latest_snapshot() -> Path | None:
    cands = sorted((p for p in SNAPSHOT_ROOT.iterdir() if p.is_dir()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> None:
    snap = _latest_snapshot()
    if snap is None:
        raise SystemExit(f'no snapshot dir under {SNAPSHOT_ROOT}')
    base = json.loads((snap / 'base_outputs.json').read_text(encoding='utf-8'))
    adap = json.loads((snap / 'adapter_outputs.json').read_text(encoding='utf-8'))
    n = len(base)
    print(f'[plot] snapshot={snap.name}  n={n}')

    base_s = [_score(r['completion']) for r in base]
    adap_s = [_score(r['completion']) for r in adap]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Panel 1: structural rate bars
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = ['opens "The stimulus"', 'TRIBE-v2 disclaimer', '"not a diagnostic"', 'peak time (seconds)']
    keys = ['opens_with_template', 'has_tribe_disclaimer', 'has_not_diagnostic', 'mentions_peak_time']
    b = [sum(float(s[k]) for s in base_s) / n for k in keys]
    a = [sum(float(s[k]) for s in adap_s) / n for k in keys]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, b, w, label='base', color='#9ecae1', edgecolor='#3182bd')
    ax.bar(x + w/2, a, w, label='brain-v2 adapter', color='#fdae6b', edgecolor='#e6550d')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha='right')
    ax.set_ylim(0, 1.05); ax.set_ylabel('fraction of outputs')
    ax.set_title(f'Structural markers — held-out n={n}')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'n30_style_transfer_bars.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'n30_style_transfer_bars.svg', bbox_inches='tight')
    plt.close(fig)

    # Panel 2: per-sample heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    grid_b = np.array([[int(s[k]) for k in keys] for s in base_s])
    grid_a = np.array([[int(s[k]) for k in keys] for s in adap_s])
    for ax, grid, title in [(axes[0], grid_b, 'base'), (axes[1], grid_a, 'brain-v2 adapter')]:
        ax.imshow(grid, aspect='auto', cmap='Oranges', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_title(title)
    axes[0].set_ylabel(f'held-out prompt index (n={n})')
    fig.suptitle('Per-sample presence of target style markers', y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'n30_per_sample.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'n30_per_sample.svg', bbox_inches='tight')
    plt.close(fig)

    print(f'[plot] wrote n30_style_transfer_bars.{{png,svg}} and n30_per_sample.{{png,svg}} under {OUT_DIR}')


if __name__ == '__main__':
    main()
