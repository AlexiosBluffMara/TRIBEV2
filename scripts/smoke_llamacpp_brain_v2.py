"""Smoke-test the v2 brain LoRA running on llama-server (bypasses Ollama).

Assumes llama-server was launched as:
    llama-server.exe -m <base.gguf> --lora brain-v2-r32-lora-f16.gguf -ngl 99 -c 2048 --port 8899

Hits /v1/chat/completions with N held-out prompts and prints results.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/smoke_llamacpp_brain_v2.py --n 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


DATASET = Path('D:/research/datasets/brain_narrations_combined_1k.jsonl')
URL     = 'http://127.0.0.1:8899/v1/chat/completions'

_SYSTEM = (
    "You are a neuroscience narration assistant. Given a stimulus description, duration, "
    "peak-activity time, and top Schaefer-400 cortical regions by mean |z|, explain what "
    "the brain is doing in 3-5 sentences. Group regions into Yeo-7 networks (Vis, SomMot, "
    "DorsAttn, SalVentAttn, Limbic, Cont, Default). Be factual, compact, and avoid "
    "diagnostic claims. End with a reminder that this is a group-averaged TRIBE v2 prediction, "
    "not a diagnostic result."
)


def _chat(user: str) -> tuple[str, float]:
    body = json.dumps({
        'messages': [
            {'role': 'system', 'content': _SYSTEM},
            {'role': 'user',   'content': user},
        ],
        'temperature': 0.4,
        'top_p': 0.9,
        'max_tokens': 400,
        'stream': False,
    }).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read().decode('utf-8'))
    return data['choices'][0]['message']['content'].strip(), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--seed', type=int, default=7777)
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.read_text(encoding='utf-8').splitlines() if l.strip()]
    rng  = random.Random(args.seed)
    picks = rng.sample(rows, min(args.n, len(rows)))
    print(f'[smoke] dataset={DATASET.name}  picks={len(picks)}  endpoint={URL}')

    for i, r in enumerate(picks):
        print('\n' + '#'*90)
        print(f'# sample {i+1}/{len(picks)}')
        print('#'*90)
        print(f'PROMPT:\n{r["prompt"][:600]}\n')
        out, dt = _chat(r['prompt'])
        print(f'OUTPUT ({dt:.1f}s):\n{out[:900]}\n')
        print(f'TARGET:\n{r["completion"][:900]}')


if __name__ == '__main__':
    main()
