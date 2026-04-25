"""Run standard LLM benchmarks via lm-evaluation-harness on base + LoRA variants.

For each (base, adapter) variant, spawns a fresh subprocess that loads the model
in 4-bit, optionally attaches a PEFT adapter, and runs lm_eval on a configurable
task list with --limit for tractable wall-clock. Subprocess isolation is
required because bitsandbytes/Unsloth state does not release cleanly via
del + torch.cuda.empty_cache (see memory feedback_eval_subprocess_vram).

Current default task list (small, high-signal, generation-compatible):
    arc_challenge  (1172 Q @ --limit applies)
    hellaswag      (10k valid — subsample heavily)
    mmlu           (multi-subject, subsample)
    gsm8k          (math, chain-of-thought)
    truthfulqa_mc1

Outputs:
    D:/research/benchmarks/<slug>/<variant-tag>/results.json
    D:/research/benchmarks/<slug>/summary.csv         (concatenated table)

Usage:
    # after overnight QLoRA runs finish, e.g.:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/run_genuine_benchmarks.py \\
        --slug gemma4-brain-bench-<ts> \\
        --variants base,r32,r64 \\
        --base-model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \\
        --r32-adapter D:/research/weights/gemma4-31b-brain-r32-<ts>/final \\
        --r64-adapter D:/research/weights/gemma4-31b-brain-r64-<ts>/final \\
        --limit 100

See also: scripts/third_party/autoresearch/ for Karpathy's autonomous
hyperparameter search pattern. This script is the static-evaluation
counterpart — known benchmarks, known variants, one run each.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_TASKS = ['arc_challenge', 'hellaswag', 'mmlu', 'gsm8k', 'truthfulqa_mc1']

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def _lm_eval_cmd(*, base: str, adapter: str | None, tasks: list[str], limit: int,
                 batch_size: int, out_path: Path, log_samples: bool,
                 cache_dir: Path | None) -> list[str]:
    model_args = [f'pretrained={base}', 'trust_remote_code=True']
    if adapter:
        model_args.append(f'peft={adapter}')
    cmd = [
        'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe',
        'D:/TRIBEV2/scripts/lm_eval_py314_wrapper.py',
        '--model', 'hf',
        '--model_args', ','.join(model_args),
        '--tasks', ','.join(tasks),
        '--batch_size', str(batch_size),
        '--output_path', str(out_path),
    ]
    if limit > 0:
        cmd += ['--limit', str(limit)]
    if log_samples:
        cmd += ['--log_samples']
    if cache_dir:
        cmd += ['--use_cache', str(cache_dir)]
    return cmd


def _run_variant(*, variant: str, base: str, adapter: str | None,
                 out_dir: Path, tasks: list[str], limit: int, batch_size: int,
                 log_samples: bool) -> dict:
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    log_path = variant_dir / 'run.log'
    cache_dir = variant_dir / 'cache'
    banner = f'[bench] variant={variant}  base={base}  adapter={adapter or "-"}\n    tasks={",".join(tasks)}  limit={limit}  bs={batch_size}\n    out={variant_dir}\n    log={log_path}'
    print(banner, flush=True)

    cmd = _lm_eval_cmd(base=base, adapter=adapter, tasks=tasks, limit=limit,
                       batch_size=batch_size, out_path=variant_dir,
                       log_samples=log_samples, cache_dir=cache_dir)
    t0 = time.time()
    with log_path.open('w', encoding='utf-8') as logf:
        logf.write(banner + '\n    cmd: ' + ' '.join(cmd) + '\n\n')
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                creationflags=CREATE_NO_WINDOW)
        rc = proc.wait()
    elapsed = time.time() - t0

    result = {'variant': variant, 'rc': rc, 'elapsed_sec': elapsed,
              'base': base, 'adapter': adapter, 'tasks': tasks, 'limit': limit}

    latest = sorted(variant_dir.glob('**/results*.json'),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if latest:
        data = json.loads(latest[0].read_text(encoding='utf-8'))
        result['results_file'] = str(latest[0])
        result['scores'] = {task: metrics for task, metrics in
                            (data.get('results') or {}).items()}
    else:
        result['scores'] = {}

    print(f'[bench] {variant} rc={rc} in {elapsed/60:.1f} min  '
          f'scores={ {k: v for k,v in list(result["scores"].items())[:3]} }',
          flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', type=str,
                    default=f'bench-{int(time.time())}')
    ap.add_argument('--out-root', type=Path,
                    default=Path('D:/research/benchmarks'))
    ap.add_argument('--base-model', type=str,
                    default='unsloth/gemma-4-31B-it-unsloth-bnb-4bit')
    ap.add_argument('--variants', type=str, required=True,
                    help='comma-separated labels from: base,r32,r64,v2,v3,cur')
    ap.add_argument('--r32-adapter', type=str, default='')
    ap.add_argument('--r64-adapter', type=str, default='')
    ap.add_argument('--v2-adapter',  type=str, default='')
    ap.add_argument('--v3-adapter',  type=str, default='')
    ap.add_argument('--cur-adapter', type=str, default='',
                    help='curriculum v4 adapter path (variant label "cur")')
    ap.add_argument('--tasks', type=str, default=','.join(DEFAULT_TASKS))
    ap.add_argument('--limit', type=int, default=100,
                    help='max samples per task (0 = full benchmark)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--log-samples', action='store_true')
    args = ap.parse_args()

    out_dir = args.out_root / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    adapter_map = {
        'base': None,
        'r32':  args.r32_adapter or None,
        'r64':  args.r64_adapter or None,
        'v2':   args.v2_adapter  or None,
        'v3':   args.v3_adapter  or None,
        'cur':  args.cur_adapter or None,
    }
    for v in variants:
        if v not in adapter_map:
            raise SystemExit(f'unknown variant: {v} (valid: base,r32,r64,v2,v3,cur)')
        if v != 'base' and not adapter_map[v]:
            raise SystemExit(f'variant {v} requires --{v}-adapter')

    print(f'[bench] out_dir={out_dir}  variants={variants}  tasks={tasks}  '
          f'limit={args.limit}  bs={args.batch_size}', flush=True)

    manifest: list[dict] = []
    for v in variants:
        res = _run_variant(variant=v, base=args.base_model,
                           adapter=adapter_map[v],
                           out_dir=out_dir, tasks=tasks, limit=args.limit,
                           batch_size=args.batch_size, log_samples=args.log_samples)
        manifest.append(res)
        (out_dir / 'manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')

    rows = []
    for m in manifest:
        for task, scores in m['scores'].items():
            row = {'variant': m['variant'], 'task': task,
                   'elapsed_min': round(m['elapsed_sec']/60, 1)}
            for metric_key, val in scores.items():
                if isinstance(val, (int, float)):
                    row[metric_key] = val
            rows.append(row)
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with (out_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'[bench] wrote summary.csv with {len(rows)} rows', flush=True)

    print('[bench] DONE.', flush=True)


if __name__ == '__main__':
    main()
