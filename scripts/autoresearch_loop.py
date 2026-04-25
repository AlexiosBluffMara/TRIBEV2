"""Multi-hour autoresearch loop for the Jemma curriculum.

Runs N experiments back-to-back. Each experiment:
  1. Picks a hypothesis (from queue.jsonl, or auto-generated from history)
  2. Builds a curriculum variant (mixture weight / filter / signal tweak)
  3. Fine-tunes an E4B adapter at smoke scale (fast: ~10-20 min)
  4. Evaluates on a lightweight three-layer eval (lm-eval limit=100 +
     tier-control n=8 + brain rubric n=15)
  5. Writes a result card + updates the leaderboard
  6. Proposes the next hypothesis based on ranked results

Designed to run unattended for multiple hours. Default wall budget is 6 h.
Crashes inside a single experiment don't abort the loop — failures get their
own result card and the next hypothesis proceeds.

Usage:
    python scripts/autoresearch_loop.py \\
        --hours 6 \\
        --root D:/research/autoresearch \\
        --base-model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit \\
        --baseline-dataset D:/research/datasets/curriculum_v4_<ts>.jsonl

The loop is tuned for idle-hour use (overnight, while you're away). It does
not attempt to run concurrently with another GPU job — it hard-checks VRAM
before each iteration and waits/backs-off if something else is active.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


PY = 'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe'
CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

DEFAULT_HYPOTHESES = [
    # Each hypothesis is a dict of overrides applied to a baseline curriculum
    # + training config. Fields:
    #   id:            unique short id
    #   name:          human-readable
    #   curriculum:    dict of build_curriculum_v4.py flag overrides
    #   training:      dict of finetune_gemma4_curriculum.py flag overrides
    #
    # Mixture weights are passed to build_curriculum_v4 as "A:x,B:y,C:z,D:w".
    {
        'id': 'h00_baseline',
        'name': 'Baseline — default mix',
        'curriculum': {'weights': 'A:1.0,B:0.8,C:0.8,D:0.7'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h01_brainheavy',
        'name': 'Brain-heavy mix (de-emphasize D)',
        'curriculum': {'weights': 'A:1.5,B:0.5,C:0.5,D:0.2'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h02_balanced',
        'name': 'Fully balanced mix',
        'curriculum': {'weights': 'A:1.0,B:1.0,C:1.0,D:1.0'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h03_rankup',
        'name': 'Rank 64 on default mix',
        'curriculum': {'weights': 'A:1.0,B:0.8,C:0.8,D:0.7'},
        'training':   {'lora-r': 64, 'lora-alpha': 128, 'lr': 2e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h04_rankdown',
        'name': 'Rank 16 (small adapter)',
        'curriculum': {'weights': 'A:1.0,B:0.8,C:0.8,D:0.7'},
        'training':   {'lora-r': 16, 'lora-alpha': 32, 'lr': 2e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h05_lrlo',
        'name': 'Lower LR, default rank',
        'curriculum': {'weights': 'A:1.0,B:0.8,C:0.8,D:0.7'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 5e-5, 'smoke-steps': 60},
    },
    {
        'id': 'h06_lrhi',
        'name': 'Higher LR, default rank',
        'curriculum': {'weights': 'A:1.0,B:0.8,C:0.8,D:0.7'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 5e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h07_tierfocus_student',
        'name': 'Student-tier heavy',
        'curriculum': {'weights': 'A:0.3,B:1.0,C:0.5,D:1.5'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h08_tierfocus_expert',
        'name': 'Expert-tier heavy (brain narration dominant)',
        'curriculum': {'weights': 'A:2.0,B:0.2,C:0.8,D:0.2'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h09_longsteps',
        'name': 'Default mix, 2x more steps',
        'curriculum': {'weights': 'A:1.0,B:0.8,C:0.8,D:0.7'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 120},
    },
    {
        'id': 'h10_rankup_balanced',
        'name': 'Rank 64 on balanced mix',
        'curriculum': {'weights': 'A:1.0,B:1.0,C:1.0,D:1.0'},
        'training':   {'lora-r': 64, 'lora-alpha': 128, 'lr': 2e-4, 'smoke-steps': 60},
    },
    {
        'id': 'h11_medheavy',
        'name': 'Medical-heavy (C dominant)',
        'curriculum': {'weights': 'A:0.5,B:0.3,C:2.0,D:0.2'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 60},
    },
    # === Neuro-domain hypotheses (post P0 pull) ============================
    {
        'id': 'h12_neuroheavy',
        'name': 'BrainGPT-heavy (neuro domain vocab dominant)',
        'curriculum': {'weights': 'A:1.0,B:2.0,C:0.6,D:0.3',
                       'braingpt-cap': '3000', 'medmcqa-cap': '500'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h13_medmcqa_only_C',
        'name': 'MedMCQA-heavy C path, no Malikeh (cleaner expert QA)',
        'curriculum': {'weights': 'A:0.8,B:0.5,C:1.8,D:0.3',
                       'medmcqa-cap': '5000', 'malikeh-cap': '0',
                       'skip-sources': 'malikeh_medqa'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h14_cochrane_transfer',
        'name': 'Cochrane tier-transfer heavy (D+expert bottomline)',
        'curriculum': {'weights': 'A:0.7,B:0.4,C:1.2,D:1.5',
                       'braingpt-cap': '500', 'medmcqa-cap': '1500'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h15_braingpt_cont',
        'name': 'BrainGPT continuation extreme (domain-LM bias)',
        'curriculum': {'weights': 'A:0.5,B:2.5,C:0.5,D:0.3',
                       'braingpt-cap': '4000'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1e-4, 'smoke-steps': 100},
    },
    {
        'id': 'h16_patient_voice_public',
        'name': 'Malikeh patient-voice C-public dominant',
        'curriculum': {'weights': 'A:0.6,B:0.5,C:2.0,D:0.5',
                       'malikeh-cap': '4000', 'medmcqa-cap': '500'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h17_neuro_rankup',
        'name': 'BrainGPT-heavy + rank 64',
        'curriculum': {'weights': 'A:1.0,B:1.8,C:0.6,D:0.3',
                       'braingpt-cap': '3000'},
        'training':   {'lora-r': 64, 'lora-alpha': 128, 'lr': 2e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h18_balanced_all_new',
        'name': 'Balanced with ALL new sources enabled (no skips)',
        'curriculum': {'weights': 'A:1.0,B:1.0,C:1.0,D:1.0'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h19_expertstack',
        'name': 'Expert stack: brain+cochrane bottomline+medmcqa',
        'curriculum': {'weights': 'A:1.5,B:0.8,C:1.5,D:0.2',
                       'medmcqa-cap': '2500', 'braingpt-cap': '1200',
                       'malikeh-cap': '500'},
        'training':   {'lora-r': 48, 'lora-alpha': 96, 'lr': 2e-4, 'smoke-steps': 80},
    },
    # === ASSET + PubMed-OA additions (post P1 pull) ========================
    {
        'id': 'h20_asset_heavy',
        'name': 'ASSET-heavy: aggressive student-tier paraphrase diversity',
        'curriculum': {'weights': 'A:0.8,B:0.5,C:0.6,D:2.2',
                       'asset-cap': '3000', 'asset-variants': '5',
                       'braingpt-cap': '500'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 2e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h21_pubmed_oa_abs',
        'name': 'PubMed-OA title→abstract synthesis (expert writing)',
        'curriculum': {'weights': 'A:0.8,B:2.2,C:0.6,D:0.3',
                       'pubmed-oa-cap': '2500', 'braingpt-cap': '800',
                       'medmcqa-cap': '800'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 90},
    },
    {
        'id': 'h22_asset_k10',
        'name': 'ASSET k=10 variants: maximum paraphrase density',
        'curriculum': {'weights': 'A:0.8,B:0.4,C:0.6,D:2.5',
                       'asset-cap': '5000', 'asset-variants': '10',
                       'braingpt-cap': '400', 'medmcqa-cap': '400'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4, 'smoke-steps': 80},
    },
    {
        'id': 'h23_fullstack',
        'name': 'Full stack: ASSET + PubMed-OA + BrainGPT all enabled',
        'curriculum': {'weights': 'A:1.0,B:1.4,C:1.0,D:1.4',
                       'asset-cap': '1500', 'asset-variants': '3',
                       'pubmed-oa-cap': '1500', 'braingpt-cap': '1500'},
        'training':   {'lora-r': 48, 'lora-alpha': 96, 'lr': 2e-4, 'smoke-steps': 100},
    },
    # --- Post-saturation hypotheses: r32 ≈ r64 at mean Δ ~+0.11, so scaling
    # rank further is zero-value. These explore orthogonal axes: alternative
    # base (h24), aggressive LR (h25), long schedule (h26), gentle LR (h27).
    {
        'id': 'h24_alt_base_26b',
        'name': 'Alternative base: Gemma-4 26B-A4B-IT (sparse mixture)',
        'curriculum': {'weights': 'A:1.0,B:0.9,C:0.9,D:0.6'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4,
                       'smoke-steps': 60},
        'base_model': 'unsloth/gemma-4-26b-a4b-it',
    },
    {
        'id': 'h25_hot_lr',
        'name': 'Aggressive LR: 4e-4 (exploit harder, saturate faster)',
        'curriculum': {'weights': 'A:1.0,B:0.9,C:0.9,D:0.6'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 4e-4,
                       'smoke-steps': 60},
    },
    {
        'id': 'h26_long_schedule',
        'name': 'Long schedule: 200 smoke steps at default LR',
        'curriculum': {'weights': 'A:1.0,B:0.9,C:0.9,D:0.6'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 1.5e-4,
                       'smoke-steps': 200},
    },
    {
        'id': 'h27_cold_lr_long',
        'name': 'Gentle LR + long schedule: 5e-5, 200 steps',
        'curriculum': {'weights': 'A:1.0,B:0.9,C:0.9,D:0.6'},
        'training':   {'lora-r': 32, 'lora-alpha': 64, 'lr': 5e-5,
                       'smoke-steps': 200},
    },
]


def _gpu_free_gb() -> float:
    """Return approximate free VRAM in GB via nvidia-smi. 0 on error."""
    try:
        si = subprocess.STARTUPINFO() if os.name == 'nt' else None
        if si is not None:
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW, startupinfo=si)
        if out.returncode != 0:
            return 0.0
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def _gpu_used_gb() -> float:
    try:
        si = subprocess.STARTUPINFO() if os.name == 'nt' else None
        if si is not None:
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW, startupinfo=si)
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def _wait_for_gpu(min_free_gb: float = 18.0, max_wait_min: int = 60) -> bool:
    """Return True when ≥ min_free_gb VRAM is free. False on timeout."""
    t0 = time.time()
    while time.time() - t0 < max_wait_min * 60:
        free = _gpu_free_gb()
        used = _gpu_used_gb()
        if free >= min_free_gb:
            print(f'[ar] GPU ready (free={free:.1f} GB, used={used:.1f} GB)', flush=True)
            return True
        print(f'[ar] GPU busy (free={free:.1f} GB, used={used:.1f} GB) — waiting…', flush=True)
        time.sleep(60)
    return False


def _run_step(cmd: list[str], log: Path, cwd: str | None = None,
              timeout_min: int = 60, stall_min: int = 20) -> int:
    """Run cmd with log streaming, hard timeout, and stall detection.

    Returns the subprocess rc, or -1 on hard timeout, or -2 on stall abort
    (no log growth for `stall_min` minutes). A stall kill fires earlier
    than the hard timeout — useful for catching frozen HF downloads without
    eating the full 90-min train window.
    """
    print(f'[ar] > {" ".join(shlex.quote(c) for c in cmd)}', flush=True)
    with log.open('w', encoding='utf-8', errors='replace') as f:
        f.write(f'# cmd: {" ".join(cmd)}\n# cwd: {cwd or os.getcwd()}\n\n')
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             cwd=cwd, creationflags=CREATE_NO_WINDOW)
        deadline = time.time() + timeout_min * 60
        last_growth_t = time.time()
        last_size = log.stat().st_size if log.exists() else 0
        while True:
            try:
                rc = p.wait(timeout=30)
                return rc
            except subprocess.TimeoutExpired:
                now = time.time()
                if now >= deadline:
                    p.kill()
                    f.write('\n\n# ABORTED: hard timeout\n')
                    return -1
                try:
                    sz = log.stat().st_size
                except OSError:
                    sz = last_size
                if sz > last_size:
                    last_size = sz
                    last_growth_t = now
                elif now - last_growth_t >= stall_min * 60:
                    p.kill()
                    f.write(f'\n\n# ABORTED: stall ({stall_min} min without '
                            f'log growth — likely frozen download)\n')
                    return -2


def _build_curriculum(hyp: dict, base_jsonl: Path | None, root: Path) -> Path | None:
    """Build a curriculum variant for hypothesis hyp. Returns output path or None.

    If `base_jsonl` is provided AND the hypothesis declares no curriculum
    overrides, reuse the baseline instead of rebuilding — saves ~1-2 min/iter."""
    cur_overrides = hyp.get('curriculum') or {}
    out = root / 'corpora' / f'{hyp["id"]}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    if base_jsonl is not None and not cur_overrides and base_jsonl.exists():
        print(f'[ar] reusing baseline {base_jsonl} (no curriculum overrides)',
              flush=True)
        try:
            import shutil as _sh
            _sh.copy2(str(base_jsonl), str(out))
            return out
        except Exception as e:
            print(f'[ar] copy failed, falling through to build: {e}', flush=True)
    args = [
        PY, 'D:/TRIBEV2/scripts/build_curriculum_v4.py',
        '--out', str(out),
        '--max-per-source', '2000',
    ]
    for k, v in cur_overrides.items():
        args.extend([f'--{k}', str(v)])
    log = root / 'logs' / f'{hyp["id"]}_01_build.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    rc = _run_step(args, log, timeout_min=15)
    if rc != 0:
        print(f'[ar] build failed (rc={rc}); see {log}', flush=True)
        return None

    check_args = [
        PY, 'D:/TRIBEV2/scripts/check_curriculum.py', str(out),
        '--samples-per-source', '0',
    ]
    check_log = root / 'logs' / f'{hyp["id"]}_01b_check.log'
    rc = _run_step(check_args, check_log, timeout_min=5)
    if rc != 0:
        print(f'[ar] curriculum validation failed (rc={rc}); skipping hyp — see '
              f'{check_log}', flush=True)
        return None
    return out


def _train_adapter(hyp: dict, data: Path, root: Path, base_model: str) -> Path | None:
    """Run smoke finetune. Returns the final/ dir on success, else None."""
    slug = f'autoresearch-{hyp["id"]}'
    tag = hyp['id']
    args = [
        PY, 'D:/TRIBEV2/scripts/finetune_gemma4_curriculum.py',
        '--dataset', str(data),
        '--base-model', base_model,
        '--slug', slug,
        '--tag', tag,
    ]
    for k, v in (hyp.get('training') or {}).items():
        args.extend([f'--{k}', str(v)])
    log = root / 'logs' / f'{hyp["id"]}_02_train.log'
    # First-iter cold start may need extra time for HF model shard download
    # (gemma-4 e4b base is ~4.3 GB sharded; slow networks can take 30-60 min
    # just to pull the shard before the first training step). Subsequent
    # iters reuse the cache so 45 min is plenty.
    rc = _run_step(args, log, timeout_min=90)
    if rc != 0:
        print(f'[ar] train failed (rc={rc}); see {log}', flush=True)
        return None
    # Adapter dir is most-recent matching prefix
    candidates = sorted(Path('D:/research/weights').glob(f'{slug}-{tag}-*'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        final = c / 'final'
        if final.exists():
            return final
    print(f'[ar] train "succeeded" but no final/ dir — see {log}', flush=True)
    return None


def _eval_lmeval(adapter: Path, base_model: str, root: Path, hyp_id: str,
                 limit: int = 100) -> dict:
    """Run lm-eval on a reduced task set. Returns flat metric dict."""
    slug = f'autoresearch-bench-{hyp_id}'
    args = [
        PY, 'D:/TRIBEV2/scripts/run_genuine_benchmarks.py',
        '--base-model', base_model,
        '--variants', 'cur',
        '--cur-adapter', str(adapter),
        '--slug', slug,
        '--limit', str(limit),
        '--tasks', 'arc_challenge,gsm8k,piqa,openbookqa',
    ]
    log = root / 'logs' / f'{hyp_id}_03_bench.log'
    rc = _run_step(args, log, timeout_min=60)
    if rc != 0:
        return {'bench_rc': rc}
    # Parse summary.csv
    summ = Path('D:/research/benchmarks') / slug / 'summary.csv'
    if not summ.exists():
        return {'bench_rc': rc, 'bench_missing_summary': True}
    out: dict = {'bench_rc': 0}
    import csv
    with summ.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            task = row.get('task', '')
            for mk in ('acc,none', 'acc_norm,none', 'exact_match,strict-match'):
                v = row.get(mk)
                if v and v not in ('None', ''):
                    try:
                        out[f'bench_{task}_{mk.split(",")[0]}'] = float(v)
                    except ValueError:
                        pass
    return out


def _eval_tier_control(adapter: Path, base_model: str, root: Path, hyp_id: str,
                       n_stimuli: int = 8) -> dict:
    out_csv = root / 'reports' / f'{hyp_id}_tier.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    args = [
        PY, 'D:/TRIBEV2/scripts/eval_tier_control.py',
        '--model', base_model,
        '--peft', str(adapter),
        '--out', str(out_csv),
        '--limit', str(n_stimuli),
        '--max-new-tokens', '200',
    ]
    log = root / 'logs' / f'{hyp_id}_04_tier.log'
    rc = _run_step(args, log, timeout_min=45)
    if rc != 0 or not out_csv.exists():
        return {'tier_rc': rc, 'tier_missing': not out_csv.exists()}
    # Aggregate FK grade + overlap per tier
    import csv, collections
    fk = collections.defaultdict(list)
    ov = collections.defaultdict(list)
    with out_csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = row['tier']
            fk[t].append(float(row['fk_grade']))
            ov[t].append(float(row['fact_overlap']))
    result = {'tier_rc': 0}
    for t in ('student', 'public', 'expert'):
        if fk[t]:
            result[f'tier_{t}_fk_median'] = sorted(fk[t])[len(fk[t]) // 2]
            result[f'tier_{t}_overlap_median'] = sorted(ov[t])[len(ov[t]) // 2]
    # FK spread (expert - student): positive means model distinguishes tiers
    if fk['expert'] and fk['student']:
        result['tier_fk_spread'] = (
            sorted(fk['expert'])[len(fk['expert']) // 2]
            - sorted(fk['student'])[len(fk['student']) // 2])
    return result


def _score_result(r: dict) -> float:
    """Single scalar ranking function. Higher is better.

    Combines: mean of bench acc_norm where available, plus tier FK spread
    (positive is good — model distinguishes tiers)."""
    acc_keys = [k for k in r if k.startswith('bench_') and k.endswith('_acc_norm')]
    acc_keys += [k for k in r if k.startswith('bench_') and k.endswith('_acc')
                 and k.replace('_acc', '_acc_norm') not in r]
    em_keys = [k for k in r if k.startswith('bench_') and k.endswith('_exact_match')]
    bench_vals = [r[k] for k in acc_keys + em_keys if isinstance(r.get(k), (int, float))]
    bench_mean = sum(bench_vals) / max(1, len(bench_vals))
    spread = max(0.0, r.get('tier_fk_spread', 0.0))
    # bench carries most weight; tier differentiation adds up to +0.05
    return bench_mean + min(0.05, spread * 0.01)


def _propose_next(history: list[dict], seen_ids: set[str]) -> dict | None:
    """Given the run history, propose a new hypothesis. Simple strategy:
    take the top-scoring id and perturb one axis that wasn't explored."""
    if not history:
        return None
    ranked = sorted(history, key=lambda h: _score_result(h), reverse=True)
    top = ranked[0]['hypothesis']
    # Generate a few perturbations of the top hypothesis
    perturbations: list[dict] = []
    base_r = int(top.get('training', {}).get('lora-r', 32))
    base_alpha = int(top.get('training', {}).get('lora-alpha', base_r * 2))
    base_lr = float(top.get('training', {}).get('lr', 2e-4))
    base_weights = top.get('curriculum', {}).get('weights', 'A:1.0,B:0.8,C:0.8,D:0.7')
    base_steps = int(top.get('training', {}).get('smoke-steps', 60))

    # Perturb LR up and down
    for scale, name in ((0.5, 'lr_half'), (2.0, 'lr_double')):
        perturbations.append({
            'id': f'{top["id"]}_{name}',
            'name': f'Derived from {top["id"]}: LR × {scale}',
            'curriculum': {'weights': base_weights},
            'training': {'lora-r': base_r, 'lora-alpha': base_alpha,
                         'lr': base_lr * scale, 'smoke-steps': base_steps},
        })
    # Perturb rank
    for r_new, name in ((base_r * 2, 'rank_up'), (max(8, base_r // 2), 'rank_down')):
        perturbations.append({
            'id': f'{top["id"]}_{name}',
            'name': f'Derived from {top["id"]}: rank {r_new}',
            'curriculum': {'weights': base_weights},
            'training': {'lora-r': r_new, 'lora-alpha': r_new * 2,
                         'lr': base_lr, 'smoke-steps': base_steps},
        })
    # More steps
    perturbations.append({
        'id': f'{top["id"]}_longer',
        'name': f'Derived from {top["id"]}: 2x smoke steps',
        'curriculum': {'weights': base_weights},
        'training': {'lora-r': base_r, 'lora-alpha': base_alpha,
                     'lr': base_lr, 'smoke-steps': base_steps * 2},
    })
    for p in perturbations:
        if p['id'] not in seen_ids:
            return p
    return None


def _save_result_card(root: Path, hyp: dict, metrics: dict, elapsed_s: float) -> None:
    card = {
        'hypothesis': hyp,
        'elapsed_s': elapsed_s,
        'ts': int(time.time()),
        **metrics,
    }
    card['score'] = _score_result(card)
    out = root / 'results' / f'{hyp["id"]}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2), encoding='utf-8')


def _load_history(root: Path) -> list[dict]:
    results = sorted((root / 'results').glob('*.json'))
    out = []
    for p in results:
        try:
            out.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            pass
    return out


def _load_bench_deltas() -> dict[str, dict]:
    """Best-effort: read bench summary.csv files and return per-hypothesis
    {mean_delta, sig_pos, sig_neg, n_tasks} for the autoresearch-bench-* slugs.

    Returns {} if compile_bench_table helpers aren't importable or no bench
    data is on disk yet.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from compile_bench_table import (  # noqa: E402
            _load_summaries, _group_by_variant, _delta_z,
        )
    except Exception:
        return {}

    bench_root = Path('D:/research/benchmarks')
    rows = _load_summaries(bench_root)
    if not rows:
        return {}
    grouped = _group_by_variant(rows)

    base_pair: tuple[str, dict] | None = None
    for (slug, variant), vals in grouped.items():
        if variant != 'base':
            continue
        if not (slug.startswith('gemma4') or 'gemma-4' in slug):
            continue
        if base_pair is None or slug > base_pair[0]:
            base_pair = (slug, vals)
    if base_pair is None:
        return {}
    _, base_vals = base_pair

    out: dict[str, dict] = {}
    prefix = 'autoresearch-bench-'
    for (slug, variant), vals in grouped.items():
        if not slug.startswith(prefix) or variant == 'base':
            continue
        hyp_id = slug[len(prefix):]
        tasks = sorted(set(vals.keys()) & set(base_vals.keys()))
        deltas: list[float] = []
        sig_pos = sig_neg = 0
        for t in tasks:
            av, ase = vals[t]
            bv, bse = base_vals[t]
            deltas.append(av - bv)
            z = _delta_z(av, ase, bv, bse)
            if z is None:
                continue
            if z >= 1.0:
                sig_pos += 1
            elif z <= -1.0:
                sig_neg += 1
        if deltas:
            out[hyp_id] = {
                'mean_delta': sum(deltas) / len(deltas),
                'sig_pos': sig_pos, 'sig_neg': sig_neg,
                'n_tasks': len(deltas),
            }
    return out


def _update_leaderboard(root: Path) -> None:
    history = _load_history(root)
    history.sort(key=lambda h: _score_result(h), reverse=True)
    deltas = _load_bench_deltas()
    lines = [
        '# Autoresearch leaderboard',
        '',
        f'_Updated {time.strftime("%Y-%m-%d %H:%M:%S")}  ·  {len(history)} experiments_',
        '',
        ('| rank | id | name | score | bench_mean | Δ base | sig+/− | '
         'fk_spread | t (min) |'),
        ('|------|----|------|------:|-----------:|------:|:------:|'
         '----------:|--------:|'),
    ]
    for i, h in enumerate(history):
        acc_keys = [k for k in h if k.startswith('bench_') and k.endswith('_acc_norm')]
        acc_keys += [k for k in h if k.startswith('bench_') and k.endswith('_acc')
                     and k.replace('_acc', '_acc_norm') not in h]
        em_keys = [k for k in h if k.startswith('bench_') and k.endswith('_exact_match')]
        bench_vals = [h[k] for k in acc_keys + em_keys
                      if isinstance(h.get(k), (int, float))]
        bench_mean = sum(bench_vals) / max(1, len(bench_vals)) if bench_vals else 0.0
        d = deltas.get(h['hypothesis']['id'])
        if d is None:
            delta_col = '—'
            sig_col = '—'
        else:
            md = d['mean_delta']
            delta_col = f'{md:+.4f}'
            sig_col = f'{d["sig_pos"]}/{d["sig_neg"]}'
        lines.append(
            f'| {i+1} | {h["hypothesis"]["id"]} | {h["hypothesis"]["name"]} | '
            f'{_score_result(h):.4f} | {bench_mean:.4f} | '
            f'{delta_col} | {sig_col} | '
            f'{h.get("tier_fk_spread", 0):+.2f} | '
            f'{h.get("elapsed_s", 0) / 60:.1f} |'
        )
    (root / 'LEADERBOARD.md').write_text('\n'.join(lines), encoding='utf-8')


_KNOWN_CURR_FLAGS = {
    'max-per-source', 'braingpt-cap', 'malikeh-cap', 'medmcqa-cap',
    'asset-cap', 'asset-variants', 'pubmed-oa-cap', 'weights',
    'skip-sources', 'only-sources', 'out',
}
_KNOWN_TRAIN_FLAGS = {
    'dataset', 'base-model', 'epochs', 'seed', 'lora-r', 'lora-alpha',
    'max-seq-length', 'batch-size', 'grad-accum', 'lr', 'tag', 'slug',
    'smoke-steps',
}


def _validate_queue() -> None:
    """Walk DEFAULT_HYPOTHESES + any queue.jsonl and flag structural issues.

    This is cheap — no GPU, no network, no disk IO beyond reading the queue.
    Catches dict-key typos, bad flag names, duplicate ids, etc. before a
    multi-hour run wastes time hitting the error mid-stream.
    """
    queued: list[dict] = list(DEFAULT_HYPOTHESES)
    queue_file = Path('D:/research/autoresearch/queue.jsonl')
    if queue_file.exists():
        with queue_file.open(encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    queued.append(json.loads(line))
                except Exception as e:
                    print(f'[validate] FATAL queue.jsonl:{i} unparseable: {e}')
                    sys.exit(2)

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for i, hyp in enumerate(queued):
        if not isinstance(hyp, dict):
            errors.append(f'#{i}: not a dict')
            continue
        hid = hyp.get('id', '')
        if not hid:
            errors.append(f'#{i}: missing id')
            continue
        if hid in seen_ids:
            errors.append(f'#{i} {hid}: duplicate id')
        seen_ids.add(hid)
        if not hyp.get('name'):
            warnings.append(f'{hid}: missing name')
        cu = hyp.get('curriculum') or {}
        if not isinstance(cu, dict):
            errors.append(f'{hid}: curriculum is not a dict')
            continue
        for k in cu:
            if k not in _KNOWN_CURR_FLAGS:
                warnings.append(f'{hid}: curriculum flag "{k}" is not in '
                                f'_KNOWN_CURR_FLAGS (typo?)')
        tr = hyp.get('training') or {}
        if not isinstance(tr, dict):
            errors.append(f'{hid}: training is not a dict')
            continue
        for k in tr:
            if k not in _KNOWN_TRAIN_FLAGS:
                warnings.append(f'{hid}: training flag "{k}" is not in '
                                f'_KNOWN_TRAIN_FLAGS (typo?)')
        bm = hyp.get('base_model')
        if bm is not None and not isinstance(bm, str):
            errors.append(f'{hid}: base_model must be str or None')

    print(f'[validate] {len(queued)} hypotheses')
    print(f'[validate]   unique ids: {len(seen_ids)}')
    if warnings:
        print(f'[validate] WARNINGS ({len(warnings)}):')
        for w in warnings:
            print(f'  ! {w}')
    if errors:
        print(f'[validate] ERRORS ({len(errors)}):')
        for e in errors:
            print(f'  x {e}')
        sys.exit(1)
    print('[validate] OK')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('D:/research/autoresearch'))
    ap.add_argument('--baseline-dataset', type=Path, default=None,
                    help='optional pre-built curriculum jsonl to reuse (reduces build step cost)')
    ap.add_argument('--base-model', type=str,
                    default='unsloth/gemma-4-e4b-it-unsloth-bnb-4bit')
    ap.add_argument('--hours', type=float, default=6.0,
                    help='wall-clock budget; stop launching new iterations after this')
    ap.add_argument('--max-iterations', type=int, default=40)
    ap.add_argument('--min-gpu-free-gb', type=float, default=18.0)
    ap.add_argument('--initial-gpu-wait-min', type=int, default=180,
                    help='how long to wait for the GPU before the first iter '
                         '(allows running while a manual bench is ongoing)')
    ap.add_argument('--bench-limit', type=int, default=100)
    ap.add_argument('--tier-n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--order', choices=['default', 'newest-first', 'shuffle'],
                    default='default',
                    help="queue order. 'newest-first' runs higher-numbered "
                         "hypothesis ids (e.g. h20_*) before lower ones, so "
                         "new-source experiments get data before budget runs out")
    ap.add_argument('--validate', action='store_true',
                    help='check all hypotheses for structural issues and exit '
                         '(no GPU, no network)')
    args = ap.parse_args()

    if args.validate:
        return _validate_queue()

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'logs').mkdir(exist_ok=True)
    (args.root / 'results').mkdir(exist_ok=True)
    (args.root / 'reports').mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    budget_s = args.hours * 3600
    t0 = time.time()

    # Seed the hypothesis queue with defaults + anything the user dropped into
    # `<root>/queue.jsonl`
    queued: list[dict] = list(DEFAULT_HYPOTHESES)
    queue_file = args.root / 'queue.jsonl'
    if queue_file.exists():
        with queue_file.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    queued.append(json.loads(line))

    # Apply order preference. `newest-first` sorts by numeric id so h20_* runs
    # before h00_*; keeps h00_baseline pinned to position 0 as a control.
    if args.order == 'newest-first':
        def _numkey(h: dict) -> int:
            hid = h.get('id', '')
            try:
                return int(hid[1:].split('_', 1)[0])
            except (ValueError, IndexError):
                return -1
        baseline = [h for h in queued if h.get('id', '').startswith('h00')]
        rest = [h for h in queued if not h.get('id', '').startswith('h00')]
        rest.sort(key=_numkey, reverse=True)
        queued = baseline + rest
        print(f'[ar] order=newest-first; head = '
              f'{[h.get("id") for h in queued[:6]]} …', flush=True)
    elif args.order == 'shuffle':
        # keep baseline as position 0; shuffle the rest
        baseline = [h for h in queued if h.get('id', '').startswith('h00')]
        rest = [h for h in queued if not h.get('id', '').startswith('h00')]
        rng.shuffle(rest)
        queued = baseline + rest
        print(f'[ar] order=shuffle; head = '
              f'{[h.get("id") for h in queued[:6]]} …', flush=True)

    seen = {Path(p).stem for p in (args.root / 'results').glob('*.json')}

    print(f'[ar] loop started; wall budget = {args.hours:.1f} h, '
          f'max iterations = {args.max_iterations}, {len(queued)} queued', flush=True)
    print(f'[ar] root = {args.root}', flush=True)

    iterations = 0
    first_iter = True
    while iterations < args.max_iterations:
        elapsed = time.time() - t0
        remaining = budget_s - elapsed
        if remaining <= 0:
            print(f'[ar] wall budget exhausted after {iterations} iterations', flush=True)
            break

        # Find the next hypothesis
        hyp = None
        while queued:
            candidate = queued.pop(0)
            if candidate['id'] not in seen:
                hyp = candidate
                break
        if hyp is None:
            hyp = _propose_next(_load_history(args.root), seen)
            if hyp is None:
                print('[ar] queue drained and no more perturbations to propose', flush=True)
                break

        seen.add(hyp['id'])
        iterations += 1
        print(f'\n[ar] === iter {iterations}: {hyp["id"]} — {hyp["name"]} ===',
              flush=True)
        print(f'[ar] elapsed {elapsed/60:.1f} min / '
              f'{args.hours*60:.0f} min  ({remaining/60:.1f} min remaining)',
              flush=True)

        # On the very first iteration, tolerate up to initial_gpu_wait_min of
        # upstream job (e.g. a manual bench that's still running). After that,
        # fall back to the per-iteration budget (default 30 min).
        wait_budget = args.initial_gpu_wait_min if first_iter else 30
        first_iter = False
        if not _wait_for_gpu(args.min_gpu_free_gb, max_wait_min=wait_budget):
            print(f'[ar] GPU not free within {wait_budget} min — aborting loop',
                  flush=True)
            break

        t_it = time.time()
        metrics: dict = {}

        # 1. build
        data = _build_curriculum(hyp, args.baseline_dataset, args.root)
        if data is None:
            metrics['phase_failed'] = 'build'
            _save_result_card(args.root, hyp, metrics, time.time() - t_it)
            _update_leaderboard(args.root)
            continue

        # 2. train — hypothesis can override base model (e.g. h24 tries 26b-a4b)
        hyp_base_model = hyp.get('base_model') or args.base_model
        adapter = _train_adapter(hyp, data, args.root, hyp_base_model)
        if adapter is None:
            metrics['phase_failed'] = 'train'
            _save_result_card(args.root, hyp, metrics, time.time() - t_it)
            _update_leaderboard(args.root)
            continue

        # 3. bench — match base to what the adapter was trained on
        if not _wait_for_gpu(args.min_gpu_free_gb, max_wait_min=10):
            metrics['phase_failed'] = 'gpu_before_bench'
            _save_result_card(args.root, hyp, metrics, time.time() - t_it)
            _update_leaderboard(args.root)
            continue
        metrics.update(_eval_lmeval(adapter, hyp_base_model, args.root,
                                     hyp['id'], args.bench_limit))

        # 4. tier-control — same base as train/bench
        if not _wait_for_gpu(args.min_gpu_free_gb, max_wait_min=10):
            metrics['phase_failed'] = 'gpu_before_tier'
            _save_result_card(args.root, hyp, metrics, time.time() - t_it)
            _update_leaderboard(args.root)
            continue
        metrics.update(_eval_tier_control(adapter, hyp_base_model, args.root,
                                           hyp['id'], args.tier_n))

        elapsed_it = time.time() - t_it
        _save_result_card(args.root, hyp, metrics, elapsed_it)
        _update_leaderboard(args.root)
        print(f'[ar] iter done in {elapsed_it/60:.1f} min  '
              f'score={_score_result({"hypothesis": hyp, **metrics}):.4f}',
              flush=True)

        # Housekeeping: keep only the latest 10 adapter dirs to avoid disk sprawl
        adapters_dir = Path('D:/research/weights')
        auto_adapters = sorted(
            adapters_dir.glob('autoresearch-*'),
            key=lambda p: p.stat().st_mtime, reverse=True)
        for extra in auto_adapters[10:]:
            try:
                shutil.rmtree(extra)
                print(f'[ar]     pruned old adapter {extra.name}', flush=True)
            except Exception as e:
                print(f'[ar]     prune fail {extra.name}: {e}', flush=True)

    print(f'\n[ar] LOOP COMPLETE: {iterations} iterations in '
          f'{(time.time() - t0)/3600:.2f} h', flush=True)
    print(f'[ar] leaderboard: {args.root / "LEADERBOARD.md"}', flush=True)


if __name__ == '__main__':
    main()
