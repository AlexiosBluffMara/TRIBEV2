"""Command builders for each task kind + resolvers for dynamic deps.

These are the only places that know the on-disk layout and script
signatures. The scheduler is agnostic — it just runs subprocesses.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .tasks import Task, TaskKind

if TYPE_CHECKING:
    from .state import State


PY = 'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe'
SCRIPTS = Path('D:/TRIBEV2/scripts')
WEIGHTS_ROOT = Path('D:/research/weights')

_PREFLIGHT_SCRIPT = '''
import os, sys
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "D:/unsloth/hf_cache"))
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id=sys.argv[1],
    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.bin"],
    resume_download=True,
)
print(f"[preflight] cached at {p}")
'''.strip()


def preflight_task(id_: str, repo_id: str, log_dir: Path) -> Task:
    """Warm the HF cache for `repo_id`. Safe to run while other tasks run."""
    cmd = [PY, '-c', _PREFLIGHT_SCRIPT, repo_id]
    return Task(
        id=id_, kind=TaskKind.PREFLIGHT, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=60, stall_min=15,
        meta={'repo_id': repo_id},
    )


def curriculum_build_task(id_: str, out: Path, weights: str, log_dir: Path,
                          max_per_source: int = 2000,
                          extra_flags: list[str] | None = None) -> Task:
    cmd = [PY, str(SCRIPTS / 'build_curriculum_v4.py'),
           '--out', str(out),
           '--max-per-source', str(max_per_source),
           '--weights', weights]
    if extra_flags:
        cmd.extend(extra_flags)
    return Task(
        id=id_, kind=TaskKind.CURRICULUM_BUILD, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=25, stall_min=10,
        meta={'out': str(out), 'weights': weights},
    )


def curriculum_check_task(id_: str, dataset: Path, deps: list[str],
                          log_dir: Path) -> Task:
    cmd = [PY, str(SCRIPTS / 'check_curriculum.py'),
           str(dataset), '--samples-per-source', '0']
    return Task(
        id=id_, kind=TaskKind.CURRICULUM_CHECK, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=5, stall_min=3,
        deps=deps,
    )


def quick_sanity_task(id_: str, base_model: str, log_dir: Path,
                      limit: int = 20, deps: list[str] | None = None) -> Task:
    """Fast GPU gate: load the base and run a tiny eval slice. ~60–90 s.

    If THIS fails, every downstream GPU task will fail. Cheap to detect
    things like 'base model path wrong' or 'bnb install broken' before
    queuing up 90 minutes of 31B training.
    """
    cmd = [PY, str(SCRIPTS / 'run_genuine_benchmarks.py'),
           '--base-model', base_model,
           '--slug', f'sanity_{id_}',
           '--limit', str(limit),
           '--tasks', 'arc_challenge',
           '--variants', 'base']
    return Task(
        id=id_, kind=TaskKind.QUICK_SANITY, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=12, stall_min=5,
        gpu_min_free_gb=10.0,
        deps=deps or [],
        meta={'base_model': base_model},
    )


def _smoke_oom_fallback(orig: Task) -> Task:
    """Halve LoRA rank/alpha on OOM retry."""
    new_cmd = list(orig.cmd)
    try:
        i = new_cmd.index('--lora-r')
        r = max(8, int(new_cmd[i + 1]) // 2)
        new_cmd[i + 1] = str(r)
    except (ValueError, IndexError):
        r = None
    try:
        j = new_cmd.index('--lora-alpha')
        a = max(16, int(new_cmd[j + 1]) // 2)
        new_cmd[j + 1] = str(a)
    except (ValueError, IndexError):
        a = None
    return Task(
        id=orig.id, kind=orig.kind, cmd=new_cmd, log=orig.log,
        timeout_min=orig.timeout_min, stall_min=orig.stall_min,
        gpu_min_free_gb=max(10.0, orig.gpu_min_free_gb - 4),
        meta={**orig.meta, 'oom_fallback_r': r, 'oom_fallback_alpha': a},
    )


def smoke_train_task(id_: str, dataset: Path, base_model: str,
                     slug: str, tag: str, log_dir: Path,
                     lora_r: int = 32, lora_alpha: int = 64,
                     lr: float = 2e-4, smoke_steps: int = 60,
                     deps: list[str] | None = None) -> Task:
    cmd = [PY, str(SCRIPTS / 'finetune_gemma4_curriculum.py'),
           '--dataset', str(dataset),
           '--base-model', base_model,
           '--slug', slug, '--tag', tag,
           '--lora-r', str(lora_r), '--lora-alpha', str(lora_alpha),
           '--lr', str(lr), '--smoke-steps', str(smoke_steps)]
    return Task(
        id=id_, kind=TaskKind.SMOKE_TRAIN, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=90, stall_min=25,
        gpu_min_free_gb=14.0,
        deps=deps or [],
        fallback_factory=_smoke_oom_fallback,
        meta={'slug': slug, 'tag': tag, 'base_model': base_model},
    )


def full_train_task(id_: str, dataset: Path, base_model: str,
                    slug: str, tag: str, log_dir: Path,
                    lora_r: int = 64, lora_alpha: int = 128,
                    lr: float = 2e-4, epochs: int = 2,
                    deps: list[str] | None = None) -> Task:
    cmd = [PY, str(SCRIPTS / 'finetune_gemma4_curriculum.py'),
           '--dataset', str(dataset),
           '--base-model', base_model,
           '--slug', slug, '--tag', tag,
           '--lora-r', str(lora_r), '--lora-alpha', str(lora_alpha),
           '--lr', str(lr), '--epochs', str(epochs)]
    return Task(
        id=id_, kind=TaskKind.FULL_TRAIN, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=180, stall_min=30,
        gpu_min_free_gb=22.0,
        deps=deps or [],
        fallback_factory=_smoke_oom_fallback,
        meta={'slug': slug, 'tag': tag, 'base_model': base_model},
    )


def _make_adapter_resolver(train_id: str, flag: str):
    """Build a resolver that injects `flag <adapter_path>` from a train dep.

    Shared by bench (flag='--cur-adapter') and tier_eval (flag='--peft').
    The train_id + flag are also persisted in task.meta so State.load can
    reconstruct this callback after resume (callbacks don't survive JSON).
    """

    def resolve(task: Task, state: 'State') -> list[str] | None:
        train = state.get(train_id)
        if train is None:
            return None
        slug = train.meta.get('slug')
        tag = train.meta.get('tag')
        if not slug or not tag:
            return None
        candidates = sorted(
            WEIGHTS_ROOT.glob(f'{slug}-{tag}-*'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        adapter = candidates[0] / 'final'
        if not adapter.exists():
            return None
        cmd = list(task.cmd)
        if flag in cmd:
            i = cmd.index(flag)
            del cmd[i:i + 2]
        cmd.extend([flag, str(adapter)])
        return cmd

    return resolve


def rebuild_resolver_from_meta(meta: dict):
    """Used by State.load: if meta has resolve_* keys, rebuild the callback."""
    tid = meta.get('resolve_from_train')
    flag = meta.get('resolve_flag')
    if tid and flag:
        return _make_adapter_resolver(tid, flag)
    return None


# Back-compat aliases — old names kept so anything imported elsewhere still works.
def _resolve_adapter_cmd(train_id: str):
    return _make_adapter_resolver(train_id, '--cur-adapter')


def _resolve_tier_adapter(train_id: str):
    return _make_adapter_resolver(train_id, '--peft')


def bench_task(id_: str, base_model: str, slug: str, log_dir: Path,
               adapter: Path | None = None, train_id: str | None = None,
               limit: int = 100,
               tasks_list: str = 'arc_challenge,gsm8k,piqa,openbookqa',
               gpu_min_free_gb: float = 14.0,
               deps: list[str] | None = None) -> Task:
    """Bench either a concrete adapter path OR defer resolution to train_id.

    `gpu_min_free_gb` should be tuned to the base model size:
        E4B (4-bit):  ~5 GB real, pass 5.0 for aggressive packing
        31B (4-bit):  ~18 GB real, keep default 14 or raise to 20
    """
    cmd = [PY, str(SCRIPTS / 'run_genuine_benchmarks.py'),
           '--base-model', base_model,
           '--slug', slug,
           '--limit', str(limit),
           '--tasks', tasks_list]
    meta_extra: dict = {}
    if adapter is not None:
        cmd.extend(['--variants', 'cur', '--cur-adapter', str(adapter)])
        resolver = None
    elif train_id is not None:
        cmd.extend(['--variants', 'cur'])
        resolver = _make_adapter_resolver(train_id, '--cur-adapter')
        meta_extra = {'resolve_from_train': train_id,
                      'resolve_flag': '--cur-adapter'}
    else:
        cmd.extend(['--variants', 'base'])
        resolver = None
    all_deps = list(deps or [])
    if train_id and train_id not in all_deps:
        all_deps.append(train_id)
    return Task(
        id=id_, kind=TaskKind.BENCH, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=45, stall_min=15,
        gpu_min_free_gb=gpu_min_free_gb,
        deps=all_deps,
        resolve_cmd=resolver,
        meta={'slug': slug, 'base_model': base_model,
              'adapter': str(adapter) if adapter else '',
              'train_id': train_id or '', **meta_extra},
    )


def tier_eval_task(id_: str, base_model: str, out_csv: Path, log_dir: Path,
                   adapter: Path | None = None,
                   train_id: str | None = None,
                   limit: int = 20, max_new_tokens: int = 256,
                   gpu_min_free_gb: float = 14.0,
                   deps: list[str] | None = None) -> Task:
    cmd = [PY, str(SCRIPTS / 'eval_tier_control.py'),
           '--model', base_model,
           '--out', str(out_csv),
           '--limit', str(limit),
           '--max-new-tokens', str(max_new_tokens)]
    resolver = None
    meta_extra: dict = {}
    if adapter is not None:
        cmd.extend(['--peft', str(adapter)])
    elif train_id is not None:
        resolver = _make_adapter_resolver(train_id, '--peft')
        meta_extra = {'resolve_from_train': train_id,
                      'resolve_flag': '--peft'}
    all_deps = list(deps or [])
    if train_id and train_id not in all_deps:
        all_deps.append(train_id)
    return Task(
        id=id_, kind=TaskKind.TIER_EVAL, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=25, stall_min=10,
        gpu_min_free_gb=gpu_min_free_gb,
        deps=all_deps,
        resolve_cmd=resolver,
        meta={'base_model': base_model, 'out_csv': str(out_csv),
              'train_id': train_id or '', **meta_extra},
    )


def compile_bench_task(id_: str, log_dir: Path,
                       deps: list[str] | None = None) -> Task:
    cmd = [PY, str(SCRIPTS / 'compile_bench_table.py')]
    return Task(
        id=id_, kind=TaskKind.COMPILE, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=4, stall_min=2,
        deps=deps or [],
        meta={'produces': 'BENCH_MATRIX.md, BENCH_DELTAS.md'},
    )


def best_adapter_task(id_: str, log_dir: Path,
                      deps: list[str] | None = None) -> Task:
    cmd = [PY, str(SCRIPTS / 'best_adapter.py')]
    return Task(
        id=id_, kind=TaskKind.COMPILE, cmd=cmd,
        log=log_dir / f'{id_}.log',
        timeout_min=4, stall_min=2,
        deps=deps or [],
        meta={'produces': 'BEST_ADAPTER.md'},
    )
