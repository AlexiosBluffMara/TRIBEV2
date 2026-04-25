"""One-panel summary figure for the paper / hackathon submission.

Two stacked rows:
- Top: Binary marker rates for base / v2 / v3 (grouped bars, 4 markers).
- Bottom: v3 - v2 paired deltas with 95% bootstrap CI error bars,
  showing that every interval crosses zero (the scaling null finding).

Inputs: latest three-way snapshot under
D:/TRIBEV2/outputs/paper/eval_stats_three_way/.

Outputs: D:/TRIBEV2/outputs/paper/figures/money_figure.{png,svg}

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_money_figure.py
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
from compute_three_way_stats import _paired_block  # type: ignore

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
    base_s = [_score(r['completion']) for r in base[:n]]
    v2_s   = [_score(r['completion']) for r in v2[:n]]
    v3_s   = [_score(r['completion']) for r in v3[:n]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.5, 7), gridspec_kw={'height_ratios': [1, 1]})

    # --- TOP: binary marker rates ---
    labels = ['opens "The stimulus"', 'TRIBE-v2 disclaimer', '"not a diagnostic"', 'peak time in s']
    keys   = ['opens_with_template', 'has_tribe_disclaimer', 'has_not_diagnostic', 'mentions_peak_time']
    b  = [sum(float(s[k]) for s in base_s) / n for k in keys]
    v2r= [sum(float(s[k]) for s in v2_s)   / n for k in keys]
    v3r= [sum(float(s[k]) for s in v3_s)   / n for k in keys]

    x = np.arange(len(labels))
    w = 0.27
    ax_top.bar(x - w, b,   w, label='base',            color='#9ecae1', edgecolor='#3182bd')
    ax_top.bar(x,     v2r, w, label='brain-v2 (r=32)', color='#fdae6b', edgecolor='#e6550d')
    ax_top.bar(x + w, v3r, w, label='brain-v3 (r=64)', color='#a1d99b', edgecolor='#31a354')
    ax_top.set_xticks(x); ax_top.set_xticklabels(labels, rotation=15, ha='right')
    ax_top.set_ylim(0, 1.08); ax_top.set_ylabel('fraction of outputs')
    ax_top.set_title(f'(a)  Structural markers — held-out n={n} paired prompts', loc='left', fontsize=11)
    ax_top.legend(loc='lower right', frameon=False, fontsize=9)
    ax_top.grid(axis='y', alpha=0.3)

    # --- BOTTOM: v3 - v2 paired deltas with 95% CI ---
    v3_v_v2 = _paired_block(v3_s, v2_s)
    order = [
        ('opens_with_template_delta',   'opens "The stimulus"'),
        ('has_not_diagnostic_delta',    '"not a diagnostic"'),
        ('mentions_peak_time_delta',    'peak time in s'),
        ('yeo7_strict_delta',           'Yeo-7 strict (abs)'),
        ('yeo7_alias_delta',            'Yeo-7 alias (any)'),
        ('n_words_delta',               'word count (scaled /100)'),
        ('ttr_delta',                   'TTR (× 10)'),
    ]
    means, lows, highs, lbls = [], [], [], []
    for key, label in order:
        v = v3_v_v2[key]
        m = v['mean']; lo, hi = v['ci95']
        if key == 'n_words_delta':
            m /= 100.0; lo /= 100.0; hi /= 100.0  # scale into the binary/rate visual range
        if key == 'ttr_delta':
            m *= 10.0; lo *= 10.0; hi *= 10.0
        means.append(m); lows.append(m - lo); highs.append(hi - m); lbls.append(label)

    y = np.arange(len(lbls))
    ax_bot.errorbar(means, y, xerr=[lows, highs], fmt='o', color='#31a354',
                    ecolor='#74c476', elinewidth=2, capsize=4, markersize=7)
    ax_bot.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
    ax_bot.set_yticks(y); ax_bot.set_yticklabels(lbls)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel('v3 − v2 paired delta  (95% bootstrap CI)')
    ax_bot.set_title('(b)  Scaling null result: every 95% CI crosses zero',
                     loc='left', fontsize=11)
    ax_bot.grid(axis='x', alpha=0.3)
    ax_bot.text(0.98, 0.02, 'word count scaled /100;  TTR scaled ×10',
                transform=ax_bot.transAxes, ha='right', va='bottom',
                fontsize=8, color='#555', style='italic')

    fig.suptitle('Brain-narration QLoRA: v2 + v3 vs base on Gemma-3-27B-IT  (n = 30)',
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'money_figure.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'money_figure.svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] wrote money_figure.{{png,svg}} under {OUT_DIR}')


if __name__ == '__main__':
    main()
