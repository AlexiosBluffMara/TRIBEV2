"""Plot v2 training-corpus descriptive statistics for the paper.

Figures:
- modality breakdown (video / audio / text / other) — pie chart
- prompt length histogram (words)
- completion length histogram (words)

Reads the v2 training-ready JSONL at:
    D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/train_text.jsonl

Writes corpus_stats.{png,svg} + corpus_stats.json to D:/TRIBEV2/outputs/paper/figures/.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_v2_corpus_stats.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_COMB = Path('D:/research/datasets/brain_narrations_combined_1k.jsonl')
TRAIN_JSONL  = Path('D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/train_text.jsonl')
OUT_DIR = Path('D:/TRIBEV2/outputs/paper/figures')


_MODALITY_MAP = {
    'silent': 'video',  # "silent <X>" descriptions come from video synth
    'scene': 'video', 'video': 'video', 'clip': 'video',
    'sound': 'audio', 'audio': 'audio', 'music': 'audio', 'dialog': 'audio',
    'talking': 'audio', 'speech': 'audio', 'podcast': 'audio', 'narration': 'audio',
    'text': 'text', 'paragraph': 'text', 'sentence': 'text', 'passage': 'text',
    'printed': 'text',
}

_PROMPT_STIM_RE = re.compile(r'Stimulus:\s*(.+?)(?:\n|$)', re.IGNORECASE)


def _infer_modality(prompt: str, row: dict) -> str:
    mod = (row.get('modality') or '').strip().lower()
    if mod in ('video', 'audio', 'text'):
        return mod
    m = _PROMPT_STIM_RE.search(prompt)
    stim = (m.group(1) if m else prompt).lower()
    for k, v in _MODALITY_MAP.items():
        if k in stim:
            return v
    return 'other'


def _pick_source() -> Path:
    # combined corpus has clean prompt/completion fields; train_text is chat-template-wrapped
    if DATASET_COMB.exists():
        return DATASET_COMB
    return TRAIN_JSONL


def _words(s: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'_\-]*", s))


def main() -> None:
    src = _pick_source()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in src.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f'[corpus] src={src.name}  rows={len(rows)}')

    modalities = Counter()
    prompt_words = []
    comp_words = []
    for r in rows:
        prompt = r.get('prompt') or r.get('text') or ''
        comp = r.get('completion') or ''
        # train_text.jsonl may wrap prompt+completion; detect
        if not prompt and 'messages' in r:
            prompt = ' '.join(m.get('content', '') for m in r['messages'] if m.get('role') == 'user')
            comp   = ' '.join(m.get('content', '') for m in r['messages'] if m.get('role') == 'assistant')
        if not prompt:
            continue
        modalities[_infer_modality(prompt, r)] += 1
        prompt_words.append(_words(prompt))
        comp_words.append(_words(comp))

    fig = plt.figure(figsize=(12, 3.8))
    ax1 = fig.add_subplot(1, 3, 1)
    labels = sorted(modalities, key=lambda k: -modalities[k])
    sizes  = [modalities[k] for k in labels]
    colors = {'video': '#74c476', 'audio': '#fd8d3c', 'text': '#6baed6', 'other': '#bdbdbd'}
    pc = ax1.pie(sizes, labels=[f'{k}\n{v}' for k, v in zip(labels, sizes)],
                 colors=[colors.get(k, '#bdbdbd') for k in labels],
                 autopct='%1.0f%%', startangle=90,
                 wedgeprops=dict(edgecolor='white', linewidth=1.2))
    ax1.set_title('Modality breakdown')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(prompt_words, bins=30, color='#9ecae1', edgecolor='#3182bd')
    ax2.set_xlabel('prompt length (words)')
    ax2.set_ylabel('count')
    ax2.set_title(f'Prompts — mean {sum(prompt_words)/len(prompt_words):.0f}, median {sorted(prompt_words)[len(prompt_words)//2]}')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(comp_words, bins=30, color='#fdae6b', edgecolor='#e6550d')
    ax3.set_xlabel('completion length (words)')
    ax3.set_title(f'Completions — mean {sum(comp_words)/len(comp_words):.0f}, median {sorted(comp_words)[len(comp_words)//2]}')

    fig.suptitle(f'Brain-v2 training corpus (n={len(rows)})', y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'corpus_stats.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'corpus_stats.svg', bbox_inches='tight')
    plt.close(fig)

    summary = {
        'source': str(src),
        'n_rows': len(rows),
        'modalities': dict(modalities),
        'prompt_words': {
            'mean': sum(prompt_words) / len(prompt_words) if prompt_words else 0,
            'min':  min(prompt_words) if prompt_words else 0,
            'max':  max(prompt_words) if prompt_words else 0,
        },
        'completion_words': {
            'mean': sum(comp_words) / len(comp_words) if comp_words else 0,
            'min':  min(comp_words) if comp_words else 0,
            'max':  max(comp_words) if comp_words else 0,
        },
    }
    (OUT_DIR / 'corpus_stats.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'[corpus] wrote {OUT_DIR}/corpus_stats.{{png,svg,json}}')


if __name__ == '__main__':
    main()
