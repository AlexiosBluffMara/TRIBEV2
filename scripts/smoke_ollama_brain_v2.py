"""Smoke-test the jemma-brain-v2-27b Ollama tag.

Picks N held-out prompts from the combined 1k corpus, sends each to Ollama's
/api/generate via the local daemon, and prints prompt / base-tag / adapter-tag
side-by-side. This verifies:

1. The LoRA GGUF is loadable on top of gemma3:27b
2. Narrations track the trained template (opens with "The stimulus..." and
   closes with the TRIBE-v2 disclaimer)

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/smoke_ollama_brain_v2.py --n 3
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


DATASET   = Path('D:/research/datasets/brain_narrations_combined_1k.jsonl')
BASE_TAG  = 'gemma3:27b'
ADAP_TAG  = 'jemma-brain-v2-27b'
OLLAMA    = 'http://127.0.0.1:11434/api/generate'


def _generate(tag: str, prompt: str) -> tuple[str, float]:
    body = json.dumps({
        'model'   : tag,
        'prompt'  : prompt,
        'stream'  : False,
        'options' : {'temperature': 0.4, 'top_p': 0.9, 'num_ctx': 2048},
    }).encode('utf-8')
    req = urllib.request.Request(OLLAMA, data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read().decode('utf-8'))
    return data.get('response', '').strip(), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7777)
    ap.add_argument('--base-only', action='store_true')
    ap.add_argument('--adapter-only', action='store_true')
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.read_text(encoding='utf-8').splitlines() if l.strip()]
    rng  = random.Random(args.seed)
    picks = rng.sample(rows, min(args.n, len(rows)))
    print(f'[smoke] dataset={DATASET.name}  picks={len(picks)}')

    for i, r in enumerate(picks):
        print('\n' + '#'*90)
        print(f'# sample {i+1}/{len(picks)}')
        print('#'*90)
        print(f'PROMPT:\n{r["prompt"][:600]}\n')

        if not args.adapter_only:
            out, dt = _generate(BASE_TAG, r['prompt'])
            print(f'BASE ({BASE_TAG}, {dt:.1f}s):\n{out[:900]}\n')

        if not args.base_only:
            out, dt = _generate(ADAP_TAG, r['prompt'])
            print(f'ADAPTER ({ADAP_TAG}, {dt:.1f}s):\n{out[:900]}\n')

        print(f'TARGET:\n{r["completion"][:900]}')


if __name__ == '__main__':
    main()
