"""Task dataclass and enums for the pipeline scheduler.

A Task is a unit of work: a subprocess invocation with timeout + stall
bounds + optional GPU reservation + optional self-heal fallback factory.
The scheduler orders tasks by priority (GPU cost, heaviest first) and
runs CPU-only tasks in parallel while one GPU task holds the device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class TaskKind(str, Enum):
    PREFLIGHT = 'preflight'
    CURRICULUM_BUILD = 'curriculum_build'
    CURRICULUM_CHECK = 'curriculum_check'
    QUICK_SANITY = 'quick_sanity'
    SMOKE_TRAIN = 'smoke_train'
    FULL_TRAIN = 'full_train'
    BENCH = 'bench'
    TIER_EVAL = 'tier_eval'
    COMPILE = 'compile'


class TaskStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    SKIPPED = 'skipped'


# Expected wall-clock minutes per task kind. Scheduler uses these as
# priority keys: heavier first so the GPU stays saturated end-to-end.
GPU_COST: dict[TaskKind, int] = {
    TaskKind.FULL_TRAIN: 120,
    TaskKind.SMOKE_TRAIN: 45,
    TaskKind.BENCH: 25,
    TaskKind.TIER_EVAL: 15,
    TaskKind.QUICK_SANITY: 5,
}

CPU_KINDS: set[TaskKind] = {
    TaskKind.PREFLIGHT, TaskKind.CURRICULUM_BUILD,
    TaskKind.CURRICULUM_CHECK, TaskKind.COMPILE,
}


@dataclass
class Task:
    id: str
    kind: TaskKind
    cmd: list[str]
    log: Path
    deps: list[str] = field(default_factory=list)
    timeout_min: int = 60
    stall_min: int = 20
    gpu_min_free_gb: float = 0.0
    retries_left: int = 2
    # Optional callable(Task) -> Task, invoked on OOM to produce a
    # degraded-config retry (e.g. halve LoRA rank).
    fallback_factory: Callable[['Task'], 'Task'] | None = None
    # Optional callable(Task, State) -> list[str] | None. Invoked right
    # before spawn if deps are met; can rewrite the cmd using output from
    # completed deps (e.g. resolve adapter path from a completed train).
    resolve_cmd: Callable[['Task', Any], list[str] | None] | None = None
    meta: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    rc: int | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def is_gpu(self) -> bool:
        return self.kind not in CPU_KINDS

    def priority(self) -> int:
        return GPU_COST.get(self.kind, 0)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'kind': self.kind.value,
            'cmd': list(self.cmd),
            'log': str(self.log),
            'deps': list(self.deps),
            'timeout_min': self.timeout_min,
            'stall_min': self.stall_min,
            'gpu_min_free_gb': self.gpu_min_free_gb,
            'retries_left': self.retries_left,
            'meta': dict(self.meta),
            'status': self.status.value,
            'rc': self.rc,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
        }
