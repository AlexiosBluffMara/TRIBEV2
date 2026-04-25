"""CLI entry: `python -m pipeline <subcommand>`.

Subcommands:

  escalate     Full-fat 31B pipeline: preflight + build + check + sanity +
               FULL_TRAIN + BENCH + TIER_EVAL + compile. Single hypothesis.

  sweep        Multi-hypothesis smoke runs (E4B). For each hypothesis:
               build + check + SMOKE_TRAIN + BENCH + TIER_EVAL. Shared
               preflight + sanity + compile. GPU tasks run back-to-back
               in priority order so the GPU stays saturated.

  resume       Resume from a crashed run.state.json; RUNNING tasks are
               reset to PENDING and respawn from scratch.

  status       Pretty-print the heartbeat JSON of the latest run.

All long commands accept --dry-run to print the plan without spawning.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the scripts/ dir importable so we can steal DEFAULT_HYPOTHESES.
_SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from .runners import (  # noqa: E402
    best_adapter_task, bench_task, compile_bench_task,
    curriculum_build_task, curriculum_check_task, full_train_task,
    preflight_task, quick_sanity_task, smoke_train_task, tier_eval_task,
)
from .scheduler import Scheduler  # noqa: E402
from .state import State  # noqa: E402
from .tasks import TaskStatus  # noqa: E402


DEFAULT_BASE_E4B = 'unsloth/gemma-4-e4b-it-unsloth-bnb-4bit'
DEFAULT_BASE_31B = 'unsloth/gemma-4-31B-it-unsloth-bnb-4bit'


def _hypotheses_by_id() -> dict[str, dict]:
    from autoresearch_loop import DEFAULT_HYPOTHESES  # type: ignore
    return {h['id']: h for h in DEFAULT_HYPOTHESES}


def _print_plan(state: State) -> None:
    print(f'\n[plan] run_id={state.run_id}  {len(state.tasks)} tasks')
    print(f'[plan] {"id":<30s} {"kind":<18s} {"timeout":>7s} '
          f'{"stall":>6s} {"deps":>s}')
    for t in state.tasks:
        deps = ','.join(t.deps) if t.deps else '-'
        print(f'[plan] {t.id:<30s} {t.kind.value:<18s} '
              f'{t.timeout_min:>5d}m {t.stall_min:>4d}m  {deps}')


def _dispatch(state: State, heartbeat_path: Path,
              max_cpu: int = 2, max_gpu_concurrent: int = 3,
              gpu_safety_margin_gb: float = 1.5) -> int:
    sched = Scheduler(state, max_cpu=max_cpu,
                      max_gpu_concurrent=max_gpu_concurrent,
                      gpu_safety_margin_gb=gpu_safety_margin_gb,
                      heartbeat_path=heartbeat_path)
    sched.run()
    counts = state.summary()['status_counts']
    failed = counts.get('failed', 0)
    return 0 if failed == 0 else 1


# ---------- escalate ----------

def cmd_escalate(args) -> int:
    ts = int(time.time())

    # Early: resolve tag (honor --from-hyp) so the run_root dirname is
    # descriptive (escalate_<tag>_<ts>).
    tag = args.tag or (args.from_hyp if args.from_hyp else f'esc_{ts}')
    run_root = Path(args.root) / f'escalate_{tag}_{ts}'
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / 'logs'

    dataset = Path(f'D:/research/datasets/curriculum_v5_escalate_{ts}.jsonl')
    slug = args.slug or f'gemma4-31b-esc-{ts}'

    weights = args.weights

    # Optional: inherit weights + hyper from a named hypothesis
    if args.from_hyp:
        hyps = _hypotheses_by_id()
        if args.from_hyp not in hyps:
            print(f'[escalate] unknown --from-hyp: {args.from_hyp}')
            return 2
        hyp = hyps[args.from_hyp]
        weights = weights or hyp['curriculum'].get('weights',
                                                   'A:1.0,B:1.0,C:1.0,D:1.0')
        if not args.lora_r:
            args.lora_r = hyp['training'].get('lora-r', 64)
        if not args.lora_alpha:
            args.lora_alpha = hyp['training'].get('lora-alpha', 128)
        if not args.lr:
            args.lr = float(hyp['training'].get('lr', 2e-4))
        if not args.base_model and hyp.get('base_model'):
            args.base_model = hyp['base_model']
    if not weights:
        weights = 'A:1.0,B:1.0,C:1.0,D:1.0'
    if not args.lora_r:
        args.lora_r = 64
    if not args.lora_alpha:
        args.lora_alpha = 128
    if not args.lr:
        args.lr = 2e-4
    if not args.base_model:
        args.base_model = DEFAULT_BASE_31B

    state = State(run_root / 'run.state.json',
                  run_id=f'escalate_{tag}_{ts}')

    # Build tasks. Priority ordering in the scheduler puts FULL_TRAIN
    # first among GPU tasks, but dep graph forces build+check+sanity first.
    pre = preflight_task(f'{tag}_preflight', args.base_model, log_dir)
    build = curriculum_build_task(
        f'{tag}_build', dataset, weights, log_dir,
        max_per_source=args.max_per_source)
    check = curriculum_check_task(
        f'{tag}_check', dataset, [build.id], log_dir)
    sanity = quick_sanity_task(
        f'{tag}_sanity', args.base_model, log_dir, deps=[pre.id])
    train = full_train_task(
        f'{tag}_train', dataset, args.base_model, slug, tag, log_dir,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lr=args.lr, epochs=args.epochs,
        deps=[pre.id, check.id, sanity.id])
    bench_slug = f'{slug}-{tag}-{ts}'
    bench = bench_task(
        f'{tag}_bench', args.base_model, bench_slug, log_dir,
        train_id=train.id, limit=args.bench_limit,
        deps=[train.id])
    tier = tier_eval_task(
        f'{tag}_tier', args.base_model,
        Path(f'D:/research/evals/tier_{tag}_{ts}.csv'), log_dir,
        train_id=train.id, limit=args.tier_limit,
        deps=[train.id])
    comp = compile_bench_task(f'{tag}_compile', log_dir,
                              deps=[bench.id])
    best = best_adapter_task(f'{tag}_best', log_dir,
                             deps=[comp.id])

    for t in (pre, build, check, sanity, train, bench, tier, comp, best):
        state.add_task(t)
    state.save()

    _print_plan(state)
    print(f'\n[escalate] weights={weights} lora-r={args.lora_r} '
          f'alpha={args.lora_alpha} lr={args.lr} epochs={args.epochs}')
    print(f'[escalate] base_model={args.base_model}')
    print(f'[escalate] slug={slug} tag={tag}')
    print(f'[escalate] dataset={dataset}')
    print(f'[escalate] state={state.path}')

    if args.dry_run:
        print('[escalate] --dry-run: exiting without spawn')
        return 0

    return _dispatch(state, run_root / 'heartbeat.json',
                     max_cpu=args.max_cpu,
                     max_gpu_concurrent=args.max_gpu,
                     gpu_safety_margin_gb=args.gpu_margin)


# ---------- sweep ----------

def cmd_sweep(args) -> int:
    ts = int(time.time())
    run_root = Path(args.root) / f'sweep_{ts}'
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / 'logs'

    hyps = _hypotheses_by_id()
    ids = [x.strip() for x in args.hypotheses.split(',') if x.strip()]
    unknown = [x for x in ids if x not in hyps]
    if unknown:
        print(f'[sweep] unknown hypotheses: {unknown}')
        print(f'[sweep] available: {sorted(hyps)[:15]}...')
        return 2

    base_model = args.base_model or DEFAULT_BASE_E4B
    state = State(run_root / 'run.state.json', run_id=f'sweep_{ts}')

    pre = preflight_task('sweep_preflight', base_model, log_dir)
    state.add_task(pre)
    sanity = quick_sanity_task('sweep_sanity', base_model, log_dir,
                               deps=[pre.id])
    state.add_task(sanity)

    train_ids: list[str] = []
    bench_ids: list[str] = []
    for hyp_id in ids:
        hyp = hyps[hyp_id]
        hyp_base = hyp.get('base_model') or base_model
        cu = hyp['curriculum']
        tr = hyp['training']
        weights = cu.get('weights', 'A:1.0,B:1.0,C:1.0,D:1.0')
        extra_flags: list[str] = []
        for k, v in cu.items():
            if k == 'weights':
                continue
            extra_flags.extend([f'--{k}', str(v)])

        dataset = Path(f'D:/research/autoresearch/corpora/'
                       f'{hyp_id}_pipeline.jsonl')

        build = curriculum_build_task(
            f'{hyp_id}_build', dataset, weights, log_dir,
            max_per_source=args.max_per_source,
            extra_flags=extra_flags or None)
        check = curriculum_check_task(
            f'{hyp_id}_check', dataset, [build.id], log_dir)

        slug = f'pipeline-{hyp_id}'
        tag = hyp_id
        train = smoke_train_task(
            f'{hyp_id}_train', dataset, hyp_base, slug, tag, log_dir,
            lora_r=tr.get('lora-r', 32),
            lora_alpha=tr.get('lora-alpha', 64),
            lr=float(tr.get('lr', 2e-4)),
            smoke_steps=tr.get('smoke-steps', 60),
            deps=[pre.id, check.id, sanity.id])
        bench_slug = f'pipeline-bench-{hyp_id}'
        bench = bench_task(
            f'{hyp_id}_bench', hyp_base, bench_slug, log_dir,
            train_id=train.id, limit=args.bench_limit,
            gpu_min_free_gb=args.bench_gpu_min_free,
            deps=[train.id])
        tier = tier_eval_task(
            f'{hyp_id}_tier', hyp_base,
            Path(f'D:/research/autoresearch/reports/'
                 f'{hyp_id}_pipeline_tier.csv'), log_dir,
            train_id=train.id, limit=args.tier_limit,
            gpu_min_free_gb=args.tier_gpu_min_free,
            deps=[train.id])

        for t in (build, check, train, bench, tier):
            state.add_task(t)
        train_ids.append(train.id)
        bench_ids.append(bench.id)

    comp = compile_bench_task('sweep_compile', log_dir,
                              deps=bench_ids)
    best = best_adapter_task('sweep_best', log_dir, deps=[comp.id])
    for t in (comp, best):
        state.add_task(t)
    state.save()

    _print_plan(state)
    print(f'\n[sweep] base_model={base_model}')
    print(f'[sweep] hypotheses: {ids}')
    print(f'[sweep] state={state.path}')

    if args.dry_run:
        print('[sweep] --dry-run: exiting without spawn')
        return 0

    return _dispatch(state, run_root / 'heartbeat.json',
                     max_cpu=args.max_cpu,
                     max_gpu_concurrent=args.max_gpu,
                     gpu_safety_margin_gb=args.gpu_margin)


# ---------- resume ----------

def cmd_resume(args) -> int:
    state_path = Path(args.state_file)
    if not state_path.exists():
        print(f'[resume] not found: {state_path}')
        return 2
    state = State.load(state_path)
    heartbeat = state_path.parent / 'heartbeat.json'
    summary = state.summary()
    print(f'[resume] {state.run_id}: {summary["status_counts"]}')
    return _dispatch(state, heartbeat,
                     max_cpu=args.max_cpu,
                     max_gpu_concurrent=args.max_gpu,
                     gpu_safety_margin_gb=args.gpu_margin)


# ---------- status ----------

def cmd_status(args) -> int:
    root = Path(args.root)
    if not root.exists():
        print(f'[status] no pipeline root at {root}')
        return 0
    # Latest run dir (by mtime) that has a heartbeat.
    runs = sorted(root.glob('*/'), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    target: Path | None = None
    for r in runs:
        if (r / 'heartbeat.json').exists():
            target = r
            break
    if target is None:
        print(f'[status] no runs with heartbeat in {root}')
        return 0
    hb_path = target / 'heartbeat.json'
    try:
        hb = json.loads(hb_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[status] read fail: {e}')
        return 1
    counts = hb.get('counts', {})
    print(f'[status] {hb["run_id"]}  elapsed={hb["elapsed_min"]:.1f}m')
    print(f'[status] GPU: used={hb["gpu"]["used_gb"]:.1f}GB  '
          f'free={hb["gpu"]["free_gb"]:.1f}GB  '
          f'util={hb["gpu"]["util_pct"]:.0f}%  '
          f'temp={hb["gpu"]["temp_c"]:.0f}C')
    gpu_running = hb.get('gpu_running') or []
    if isinstance(gpu_running, str):  # legacy single-task heartbeat
        gpu_running = [gpu_running]
    gpu_cap = hb.get('gpu_cap', 1)
    gpu_conc = hb.get('gpu_concurrent', len(gpu_running))
    print(f'[status] running: gpu=[{gpu_conc}/{gpu_cap}] '
          f'{",".join(gpu_running) or "-"}  '
          f'cpu={",".join(hb.get("cpu_running") or []) or "-"}')
    print(f'[status] counts: {counts}')
    running = [t for t in hb.get('tasks', [])
               if t.get('status') == 'running']
    pending = [t for t in hb.get('tasks', [])
               if t.get('status') == 'pending']
    failed = [t for t in hb.get('tasks', [])
              if t.get('status') == 'failed']
    if running:
        print('[status] active:')
        for t in running:
            e = t.get('elapsed_min')
            print(f'  - {t["id"]:<32s} {t["kind"]:<16s} '
                  f'elapsed={e:.1f}m' if e else f'  - {t["id"]}')
    if pending:
        print(f'[status] queued ({len(pending)}): '
              f'{",".join(t["id"] for t in pending[:5])}'
              f'{"…" if len(pending) > 5 else ""}')
    if failed:
        print(f'[status] FAILED ({len(failed)}):')
        for t in failed[:5]:
            print(f'  - {t["id"]} rc={t.get("rc")}')
    return 0


# ---------- entry ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='python -m pipeline',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--root', default='D:/research/pipeline',
                    help='root dir for run artifacts')
    sub = ap.add_subparsers(dest='cmd', required=True)

    e = sub.add_parser('escalate', help='full 31B pipeline for one hypothesis')
    e.add_argument('--from-hyp', default='',
                   help='inherit weights + hyper from a named hypothesis')
    e.add_argument('--base-model', default='')
    e.add_argument('--weights', default='')
    e.add_argument('--lora-r', type=int, default=0)
    e.add_argument('--lora-alpha', type=int, default=0)
    e.add_argument('--lr', type=float, default=0.0)
    e.add_argument('--epochs', type=int, default=2)
    e.add_argument('--max-per-source', type=int, default=2000)
    e.add_argument('--bench-limit', type=int, default=200)
    e.add_argument('--tier-limit', type=int, default=20)
    e.add_argument('--slug', default='')
    e.add_argument('--tag', default='')
    e.add_argument('--max-cpu', type=int, default=2,
                   help='max parallel CPU tasks (default 2)')
    e.add_argument('--max-gpu', type=int, default=3,
                   help='max concurrent GPU tasks when VRAM allows (default 3)')
    e.add_argument('--gpu-margin', type=float, default=1.5,
                   help='VRAM safety margin in GB on top of task min (default 1.5)')
    e.add_argument('--dry-run', action='store_true')
    e.set_defaults(func=cmd_escalate)

    s = sub.add_parser('sweep', help='smoke-run multiple hypotheses back-to-back')
    s.add_argument('--hypotheses', required=True,
                   help='comma-separated hypothesis ids (e.g. h20_asset_heavy,h21_pubmed_oa_abs)')
    s.add_argument('--base-model', default='')
    s.add_argument('--max-per-source', type=int, default=2000)
    s.add_argument('--bench-limit', type=int, default=100)
    s.add_argument('--tier-limit', type=int, default=8)
    s.add_argument('--max-cpu', type=int, default=2)
    s.add_argument('--max-gpu', type=int, default=3)
    s.add_argument('--gpu-margin', type=float, default=1.5)
    s.add_argument('--bench-gpu-min-free', type=float, default=5.0,
                   help='Real bench VRAM need (GB). E4B 4-bit ~3-5; '
                        '31B 4-bit raise to 14+. (default 5.0 for E4B)')
    s.add_argument('--tier-gpu-min-free', type=float, default=5.0,
                   help='Real tier_eval VRAM need (GB). (default 5.0 for E4B)')
    s.add_argument('--dry-run', action='store_true')
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser('resume', help='resume from state.json')
    r.add_argument('state_file')
    r.add_argument('--max-cpu', type=int, default=2)
    r.add_argument('--max-gpu', type=int, default=3)
    r.add_argument('--gpu-margin', type=float, default=1.5)
    r.set_defaults(func=cmd_resume)

    st = sub.add_parser('status', help='dump heartbeat of latest run')
    st.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
