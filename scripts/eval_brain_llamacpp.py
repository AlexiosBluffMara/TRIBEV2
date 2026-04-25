"""Expanded held-out eval using llama-server (base vs adapter).

Flow:
    1. Pick N held-out prompts from the combined corpus and write picks.json.
    2. Expect a llama-server running on 8899 for base (no --lora) OR adapter (--lora),
       controlled via --phase {base,adapter}.
    3. Generate one completion per pick, write {base,adapter}_outputs.json matching the
       on-disk schema used by scripts/compute_eval_stats.py.

The launcher in the same dir (eval_brain_llamacpp_run.ps1 / .sh) is responsible for
starting/stopping llama-server. That separation keeps VRAM under control — one 27B base
at a time.

Usage (typical):
    # 1. Launch base server (no --lora) on 8899
    # 2. python scripts/eval_brain_llamacpp.py --phase base --n 30 --out-dir C:/Users/soumi/AppData/Local/Temp/eval_brain_<ts>
    # 3. Kill base server, launch adapter server
    # 4. python scripts/eval_brain_llamacpp.py --phase adapter --n 30 --out-dir <same>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

DATASET_DEFAULT = Path('D:/research/datasets/brain_narrations_combined_1k.jsonl')
URL = 'http://127.0.0.1:8899/v1/chat/completions'

_SYSTEM = (
    "You are a neuroscience narration assistant. Given a stimulus description, duration, "
    "peak-activity time, and top Schaefer-400 cortical regions by mean |z|, explain what "
    "the brain is doing in 3-5 sentences. Group regions into Yeo-7 networks (Vis, SomMot, "
    "DorsAttn, SalVentAttn, Limbic, Cont, Default). Be factual, compact, and avoid "
    "diagnostic claims. End with a reminder that this is a group-averaged TRIBE v2 prediction, "
    "not a diagnostic result."
)


def _chat(user: str, *, timeout: int = 900) -> tuple[str, float]:
    body = json.dumps({
        'messages': [
            {'role': 'system', 'content': _SYSTEM},
            {'role': 'user', 'content': user},
        ],
        'temperature': 0.4,
        'top_p': 0.9,
        'max_tokens': 400,
        'stream': False,
    }).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode('utf-8'))
    return data['choices'][0]['message']['content'].strip(), time.time() - t0


def _load_or_build_picks(picks_path: Path, dataset: Path, n: int, seed: int) -> list[dict]:
    if picks_path.exists():
        picks = json.loads(picks_path.read_text(encoding='utf-8'))
        print(f'[eval] reusing {picks_path.name} ({len(picks)} picks)')
        return picks
    rows = [json.loads(l) for l in dataset.read_text(encoding='utf-8').splitlines() if l.strip()]
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n, len(rows)))
    picks_path.write_text(json.dumps(picks, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[eval] wrote {picks_path.name} ({len(picks)} picks)')
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True,
                    help="output label; writes {phase}_outputs.json. "
                         "Conventional values: 'base', 'adapter', 'v2', 'v3'.")
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--dataset', type=Path, default=DATASET_DEFAULT)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    picks_path = args.out_dir / 'picks.json'
    out_path   = args.out_dir / f'{args.phase}_outputs.json'

    picks = _load_or_build_picks(picks_path, args.dataset, args.n, args.seed)
    print(f'[eval] phase={args.phase}  picks={len(picks)}  out={out_path}')

    results = []
    for i, p in enumerate(picks):
        try:
            out, dt = _chat(p['prompt'])
        except Exception as e:
            print(f'[eval] pick {i+1} failed: {e}')
            out, dt = f'<<ERROR: {e}>>', -1.0
        results.append({
            'prompt': p['prompt'],
            'completion': out,
            'latency_s': dt,
        })
        preview = out.replace('\n', ' ')[:100]
        print(f'  [{i+1}/{len(picks)}] {dt:6.1f}s | {preview}...')
        sys.stdout.flush()

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[eval] wrote {out_path}')


if __name__ == '__main__':
    main()
