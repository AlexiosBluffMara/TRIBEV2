"""Pick the top hypothesis from the autoresearch leaderboard and escalate it
to a full 31B curriculum finetune (plus genuine-benchmark + tier-control).

Reads `D:/research/autoresearch/results/*.json`, ranks by score, optionally
filters out hypotheses that failed at any phase, then:

  1. Re-builds the curriculum with the winning hypothesis's flags at full size
     (2000 rows/source).
  2. Fine-tunes Gemma-4-31B on that curriculum with the winning LR/rank/alpha.
  3. Runs lm-eval + tier-control on the resulting adapter.

Usage:
    python scripts/apply_winner.py                   # top-1 winner
    python scripts/apply_winner.py --top-k 3         # show top-3, ask which
    python scripts/apply_winner.py --hyp-id h12_neuroheavy
    python scripts/apply_winner.py --dry-run         # print plan, don't run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PY = 'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe'
CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def _bench_delta_vs_base(hyp_id: str, bench_root: Path) -> dict | None:
    """Mean bench Δ of an autoresearch hypothesis vs gemma4 base.

    Returns None when no matching bench summary exists on disk yet.
    Otherwise {mean_delta, sig_pos, sig_neg, base_slug, n_tasks}.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from compile_bench_table import (  # noqa: E402
            _load_summaries, _group_by_variant, _delta_z,
        )
    except Exception as e:
        print(f'[apply] bench gate unavailable ({e}); skipping')
        return None

    rows = _load_summaries(bench_root)
    if not rows:
        return None
    grouped = _group_by_variant(rows)

    ar_slug = f'autoresearch-bench-{hyp_id}'
    candidate = None
    for (slug, variant), vals in grouped.items():
        if slug == ar_slug and variant != 'base':
            candidate = vals
            break
    if candidate is None:
        return None

    base_pair: tuple[str, dict] | None = None
    for (slug, variant), vals in grouped.items():
        if variant != 'base':
            continue
        if not (slug.startswith('gemma4') or 'gemma-4' in slug):
            continue
        if base_pair is None or slug > base_pair[0]:
            base_pair = (slug, vals)
    if base_pair is None:
        return None
    base_slug, base_vals = base_pair

    tasks = sorted(set(candidate.keys()) & set(base_vals.keys()))
    deltas: list[float] = []
    sig_pos = 0
    sig_neg = 0
    for t in tasks:
        av, ase = candidate[t]
        bv, bse = base_vals[t]
        deltas.append(av - bv)
        z = _delta_z(av, ase, bv, bse)
        if z is None:
            continue
        if z >= 1.0:
            sig_pos += 1
        elif z <= -1.0:
            sig_neg += 1

    if not deltas:
        return None
    return {
        'mean_delta': sum(deltas) / len(deltas),
        'sig_pos': sig_pos,
        'sig_neg': sig_neg,
        'base_slug': base_slug,
        'n_tasks': len(deltas),
    }


def _score(h: dict) -> float:
    acc_keys = [k for k in h if k.startswith('bench_') and k.endswith('_acc_norm')]
    acc_keys += [k for k in h if k.startswith('bench_') and k.endswith('_acc')
                 and k.replace('_acc', '_acc_norm') not in h]
    em_keys = [k for k in h if k.startswith('bench_') and k.endswith('_exact_match')]
    bench_vals = [h[k] for k in acc_keys + em_keys if isinstance(h.get(k), (int, float))]
    bench_mean = sum(bench_vals) / max(1, len(bench_vals)) if bench_vals else 0.0
    spread = max(0.0, h.get('tier_fk_spread', 0.0))
    return bench_mean + min(0.05, spread * 0.01)


def _load_history(root: Path) -> list[dict]:
    results = sorted((root / 'results').glob('*.json'))
    out = []
    for p in results:
        try:
            out.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            pass
    return out


def _run(cmd: list[str], log: Path) -> int:
    print(f'\n[apply] > {" ".join(shlex.quote(c) for c in cmd)}')
    print(f'[apply]   log: {log}')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8', errors='replace') as f:
        f.write(f'# cmd: {" ".join(cmd)}\n\n')
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             creationflags=CREATE_NO_WINDOW)
        rc = p.wait()
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('D:/research/autoresearch'))
    ap.add_argument('--hyp-id', type=str, default='',
                    help='override; use this exact hypothesis id')
    ap.add_argument('--top-k', type=int, default=5,
                    help='show top-k but auto-pick top-1 (for info)')
    ap.add_argument('--slug', type=str,
                    default='gemma4-31b-v5-winner',
                    help='adapter slug for 31B train')
    ap.add_argument('--base-model', type=str,
                    default='unsloth/gemma-4-31B-it-unsloth-bnb-4bit')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--bench-limit', type=int, default=200)
    ap.add_argument('--tier-n', type=int, default=20)
    ap.add_argument('--max-per-source', type=int, default=2000)
    ap.add_argument('--quick', action='store_true',
                    help='use E4B base, 1 epoch, smaller bench/tier; ~20 min '
                         'sanity pass instead of 90+ min 31B full train')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='escalate even if winner has net-negative bench Δ '
                         'vs base (normally refused)')
    ap.add_argument('--bench-root', type=Path,
                    default=Path('D:/research/benchmarks'),
                    help='root of genuine-benchmark summary.csv files for '
                         'the bench-delta gate')
    args = ap.parse_args()

    if args.quick:
        args.base_model = 'unsloth/gemma-4-e4b-it-unsloth-bnb-4bit'
        args.slug = args.slug.replace('-31b-', '-e4b-') \
            if '-31b-' in args.slug else f'{args.slug}-e4b'
        args.epochs = max(1, args.epochs // 2)
        args.bench_limit = min(args.bench_limit, 100)
        args.tier_n = min(args.tier_n, 10)
        args.max_per_source = min(args.max_per_source, 1000)
        print(f'[apply] --quick: base={args.base_model} epochs={args.epochs} '
              f'bench_limit={args.bench_limit} tier_n={args.tier_n} '
              f'max_per_source={args.max_per_source}')

    history = _load_history(args.root)
    history = [h for h in history if not h.get('phase_failed')]
    if not history:
        print(f'[apply] no completed hypotheses in {args.root / "results"}')
        return
    history.sort(key=_score, reverse=True)

    print(f'[apply] top {min(args.top_k, len(history))} hypotheses:')
    for i, h in enumerate(history[:args.top_k]):
        hyp = h.get('hypothesis', {})
        tr = hyp.get('training', {})
        print(f'  {i+1}. {hyp.get("id"):28s}  score={_score(h):.4f}  '
              f'r={tr.get("lora-r")}  α={tr.get("lora-alpha")}  '
              f'lr={tr.get("lr")}  steps={tr.get("smoke-steps")}')

    if args.hyp_id:
        winner = next((h for h in history
                       if h.get('hypothesis', {}).get('id') == args.hyp_id), None)
        if winner is None:
            print(f'[apply] hyp-id {args.hyp_id} not found (must have completed)')
            return
    else:
        winner = history[0]

    hyp = winner['hypothesis']
    tr = hyp.get('training', {})
    cu = hyp.get('curriculum', {})
    print(f'\n[apply] WINNER: {hyp["id"]} — {hyp.get("name", "")}')
    print(f'[apply]   curriculum: {cu}')
    print(f'[apply]   training:   {tr}')
    print(f'[apply]   score:      {_score(winner):.4f}')

    # Bench-delta gate: refuse to 31B-train a hypothesis that regresses vs
    # the family base. The autoresearch score alone is noisy on a smoke run;
    # absolute accuracy vs base is the sanity check.
    delta_info = _bench_delta_vs_base(hyp['id'], args.bench_root)
    if delta_info is None:
        print(f'[apply]   bench gate: no data yet for '
              f'autoresearch-bench-{hyp["id"]} (skipping)')
    else:
        md = delta_info['mean_delta']
        print(f'[apply]   bench Δ vs {delta_info["base_slug"]}: '
              f'mean {md:+.4f}  sig+/− {delta_info["sig_pos"]}/'
              f'{delta_info["sig_neg"]}  n_tasks={delta_info["n_tasks"]}')
        if md < 0 and not args.force:
            print('')
            print(f'[apply] REFUSING: hypothesis mean Δ {md:+.4f} is '
                  'net-negative vs base.')
            print('[apply]   Escalating would likely produce a 31B adapter '
                  'worse than the base model on lm-eval.')
            print('[apply]   Options:')
            print('[apply]     --force   escalate anyway (you own the burn)')
            print('[apply]     --hyp-id  pick a specific hypothesis with '
                  'better bench')
            print('[apply]     run scripts/best_adapter.py to see the '
                  'current ESCALATE candidate (may already exist)')
            return

    ts = int(time.time())
    log_dir = Path(f'D:/research/logs/apply_winner_{ts}')
    dataset = Path(f'D:/research/datasets/curriculum_v5_winner_{ts}.jsonl')

    # 1. Build full-size curriculum with winning flags
    build_cmd = [
        PY, 'D:/TRIBEV2/scripts/build_curriculum_v4.py',
        '--out', str(dataset),
        '--max-per-source', str(args.max_per_source),
    ]
    for k, v in cu.items():
        build_cmd.extend([f'--{k}', str(v)])

    # 1b. Validate curriculum before burning GPU on it
    check_cmd = [
        PY, 'D:/TRIBEV2/scripts/check_curriculum.py', str(dataset),
        '--samples-per-source', '0',
    ]

    # 2. 31B train with winning LR/rank/alpha (ignore smoke-steps at 31B full train)
    train_cmd = [
        PY, 'D:/TRIBEV2/scripts/finetune_gemma4_curriculum.py',
        '--dataset', str(dataset),
        '--base-model', args.base_model,
        '--slug', args.slug,
        '--tag', hyp['id'],
        '--epochs', str(args.epochs),
        '--lora-r', str(tr.get('lora-r', 64)),
        '--lora-alpha', str(tr.get('lora-alpha', 128)),
        '--lr', str(tr.get('lr', 2e-4)),
    ]

    print('\n[apply] plan:')
    for c in (build_cmd, check_cmd, train_cmd):
        print(f'  $ {" ".join(c)}')
    print(f'  logs -> {log_dir}')

    if args.dry_run:
        print('[apply] --dry-run; not executing')
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    rc = _run(build_cmd, log_dir / '01_build.log')
    if rc != 0:
        print(f'[apply] build failed rc={rc}')
        return

    check_log = log_dir / '01b_check.log'
    rc = _run(check_cmd, check_log)
    if rc != 0:
        print(f'[apply] FATAL: curriculum validation failed rc={rc}')
        print(f'[apply]   refusing to burn GPU on bad dataset; see {check_log}')
        return
    try:
        check_text = check_log.read_text(encoding='utf-8', errors='replace')
    except Exception:
        check_text = ''
    if '[check] OK' not in check_text:
        print(f'[apply] FATAL: curriculum validator did not emit [check] OK')
        print(f'[apply]   see {check_log}')
        return

    rc = _run(train_cmd, log_dir / '02_train.log')
    if rc != 0:
        print(f'[apply] train failed rc={rc}')
        return

    adapters = sorted(Path('D:/research/weights').glob(f'{args.slug}-{hyp["id"]}-*'),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not adapters:
        print('[apply] no adapter produced — aborting')
        return
    final = adapters[0] / 'final'
    print(f'[apply] adapter: {final}')

    bench_slug = f'{args.slug}-{hyp["id"]}-{ts}'
    bench_cmd = [
        PY, 'D:/TRIBEV2/scripts/run_genuine_benchmarks.py',
        '--base-model', args.base_model,
        '--variants', 'cur',
        '--cur-adapter', str(final),
        '--slug', bench_slug,
        '--limit', str(args.bench_limit),
    ]
    _run(bench_cmd, log_dir / '03_bench.log')

    tier_out = Path(f'D:/research/evals/tier_winner_{hyp["id"]}_{ts}.csv')
    tier_cmd = [
        PY, 'D:/TRIBEV2/scripts/eval_tier_control.py',
        '--model', args.base_model,
        '--peft', str(final),
        '--out', str(tier_out),
        '--limit', str(args.tier_n),
        '--max-new-tokens', '256',
    ]
    _run(tier_cmd, log_dir / '04_tier.log')

    # Refresh the cross-model bench matrix so the new winner shows up.
    compile_cmd = [
        PY, 'D:/TRIBEV2/scripts/compile_bench_table.py',
    ]
    _run(compile_cmd, log_dir / '05_compile.log')

    print('\n[apply] DONE')
    print(f'[apply]   adapter: {final}')
    print(f'[apply]   bench:   D:/research/benchmarks/{bench_slug}')
    print(f'[apply]   tier:    {tier_out}')
    print(f'[apply]   logs:    {log_dir}')
    print(f'[apply]   matrix:  D:/research/benchmarks/BENCH_MATRIX.md')


if __name__ == '__main__':
    main()
