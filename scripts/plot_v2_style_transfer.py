"""Render a publication-ready bar chart of style-transfer metrics (base vs adapter).

Reads eval_stats_v2.json, emits style_transfer_bars.{png,svg}.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_v2_style_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STATS_PATH = Path('D:/TRIBEV2/outputs/paper/eval_stats/eval_stats_v2.json')
OUT_DIR    = Path('D:/TRIBEV2/outputs/paper/figures')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = json.loads(STATS_PATH.read_text(encoding='utf-8'))
    base = s['base_summary']; adap = s['adapter_summary']
    n = s['n_pairs']

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: binary style markers
    bin_labels = ['opens "The stimulus"', 'TRIBE-v2 disclaimer', 'not-diagnostic phrase', 'peak time (seconds)']
    bin_keys   = ['opens_with_template_rate', 'has_tribe_disclaimer_rate',
                  'has_not_diagnostic_rate', 'mentions_peak_time_rate']
    b_vals = [base[k] for k in bin_keys]
    a_vals = [adap[k] for k in bin_keys]
    x = np.arange(len(bin_labels))
    w = 0.38
    axL.bar(x - w/2, b_vals, w, label='base', color='#9ecae1', edgecolor='#3182bd')
    axL.bar(x + w/2, a_vals, w, label='brain-v2 adapter', color='#fdae6b', edgecolor='#e6550d')
    axL.set_xticks(x)
    axL.set_xticklabels(bin_labels, rotation=18, ha='right')
    axL.set_ylim(0, 1.05)
    axL.set_ylabel('fraction of outputs')
    axL.set_title(f'Template markers (n={n})')
    axL.legend(loc='upper left', frameon=False)
    axL.grid(axis='y', alpha=0.3)

    # Right: continuous metrics (normalized to base=1.0 for visual comparability)
    cont_labels = ['Yeo-7 abbrev. count', 'Yeo-7 any-alias count', 'word count', 'TTR (diversity)']
    cont_base = [base['yeo7_networks_mentioned_mean'], base['yeo7_any_alias_mean'],
                 base['n_words_mean'], base['ttr_mean']]
    cont_adap = [adap['yeo7_networks_mentioned_mean'], adap['yeo7_any_alias_mean'],
                 adap['n_words_mean'], adap['ttr_mean']]
    ratios = [a / b if b else 0.0 for a, b in zip(cont_adap, cont_base)]
    x2 = np.arange(len(cont_labels))
    axR.bar(x2, ratios, 0.6,
            color=['#fdae6b' if r >= 1 else '#c6dbef' for r in ratios],
            edgecolor='#636363')
    axR.axhline(1.0, color='#636363', linestyle='--', linewidth=1)
    for i, r in enumerate(ratios):
        axR.text(i, r + 0.02, f'{r:.2f}×', ha='center', fontsize=9)
    axR.set_xticks(x2)
    axR.set_xticklabels(cont_labels, rotation=18, ha='right')
    axR.set_ylabel('adapter / base')
    axR.set_title('Continuous metrics (ratio)')
    axR.grid(axis='y', alpha=0.3)

    fig.suptitle('Brain-v2 QLoRA vs Gemma-3-27B-IT base — held-out narration panel',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'style_transfer_bars.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'style_transfer_bars.svg', bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] wrote {OUT_DIR}/style_transfer_bars.{{png,svg}}')


if __name__ == '__main__':
    main()
