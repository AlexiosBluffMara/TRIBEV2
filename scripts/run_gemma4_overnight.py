"""Sequential overnight driver for Gemma-4-31B brain-narration QLoRA runs.

Launches one finetune_gemma4_brain.py invocation at a time in a subprocess,
streams stdout to a per-run log file, and moves on once each finishes. Intended
for hours-long unattended sessions on the RTX 5090 — the wrapper is kept
deliberately dumb (no retries, no parallelism) so we don't risk OOM from two
31B loads on the same GPU.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/run_gemma4_overnight.py \
        --plan r32-v2parity r64-v3parity

    # dry-run (print the invocation plan, don't launch):
    ... --dry-run

Each run's artifacts land at
    D:/research/weights/gemma4-31b-brain-{tag}-{ts}/...
and the log lives next to them as overnight.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PY        = 'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe'
SCRIPT    = 'D:/TRIBEV2/scripts/finetune_gemma4_brain.py'
DATASET   = 'D:/research/datasets/brain_narrations_combined_2189.jsonl'
LOG_ROOT  = Path('D:/research/logs/gemma4-overnight')

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


RECIPES: dict[str, list[str]] = {
    'r32-v2parity': [
        '--lora-r', '32', '--lora-alpha', '64',
        '--tag', 'r32',
        '--epochs', '3',
        '--max-seq-length', '2048',
        '--batch-size', '1', '--grad-accum', '8',
        '--lr', '1.5e-4',
    ],
    'r64-v3parity': [
        '--lora-r', '64', '--lora-alpha', '128',
        '--tag', 'r64',
        '--epochs', '3',
        '--max-seq-length', '2048',
        '--batch-size', '1', '--grad-accum', '8',
        '--lr', '1.5e-4',
    ],
    'r16-probe': [
        '--lora-r', '16', '--lora-alpha', '32',
        '--tag', 'r16',
        '--epochs', '3',
        '--max-seq-length', '2048',
        '--batch-size', '1', '--grad-accum', '8',
        '--lr', '1.5e-4',
    ],
    'smoke': [
        '--lora-r', '32', '--lora-alpha', '64',
        '--tag', 'smoke',
        '--epochs', '1',
        '--smoke-steps', '5',
        '--max-seq-length', '2048',
        '--batch-size', '1', '--grad-accum', '8',
        '--lr', '1.5e-4',
    ],
}


def _invocation(recipe: str) -> list[str]:
    if recipe not in RECIPES:
        raise SystemExit(f'unknown recipe: {recipe} (known: {sorted(RECIPES)})')
    return [PY, SCRIPT, '--dataset', DATASET] + RECIPES[recipe]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', nargs='+', default=['r32-v2parity', 'r64-v3parity'],
                    help='ordered recipe names — see RECIPES dict.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.time()

    manifest: list[dict] = []
    for i, recipe in enumerate(args.plan):
        cmd = _invocation(recipe)
        ts = int(time.time())
        log_path = LOG_ROOT / f'{ts}_{recipe}.log'
        banner = f'[overnight] [{i+1}/{len(args.plan)}] {recipe}\n    cmd: {" ".join(cmd)}\n    log: {log_path}'
        print(banner, flush=True)
        manifest.append({'recipe': recipe, 'cmd': cmd, 'log': str(log_path), 'start_ts': ts})
        if args.dry_run:
            continue

        t0 = time.time()
        with log_path.open('w', encoding='utf-8') as logf:
            logf.write(banner + '\n\n')
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=logf, stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
            rc = proc.wait()
        elapsed = time.time() - t0
        manifest[-1]['elapsed_sec'] = elapsed
        manifest[-1]['rc'] = rc
        print(f'[overnight] {recipe} finished rc={rc} in {elapsed/60:.1f} min', flush=True)

        (LOG_ROOT / 'manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8'
        )
        if rc != 0:
            print(f'[overnight] stopping plan after non-zero exit from {recipe}', flush=True)
            break

    total = time.time() - overall_t0
    print(f'[overnight] plan complete in {total/60:.1f} min')


if __name__ == '__main__':
    main()
