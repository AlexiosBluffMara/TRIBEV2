"""Persistent run state.

The scheduler writes state to a JSON file after every transition so the
run can be resumed from any point: if the Python process dies, the next
`python -m pipeline resume <state.json>` picks up exactly where it left
off (RUNNING tasks reset to PENDING so they re-dispatch from scratch).

Also keeps an append-only event log for post-mortem.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .tasks import Task, TaskKind, TaskStatus


class State:
    def __init__(self, path: Path, run_id: str | None = None):
        self.path = Path(path)
        self.run_id = run_id or f'run_{int(time.time())}'
        self.started_at = time.time()
        self.tasks: list[Task] = []
        self.events: list[dict] = []
        self._task_index: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        if task.id in self._task_index:
            raise ValueError(f'duplicate task id: {task.id}')
        self.tasks.append(task)
        self._task_index[task.id] = task

    def get(self, task_id: str) -> Task | None:
        return self._task_index.get(task_id)

    def add_event(self, task_id: str, kind: str, data: dict) -> None:
        self.events.append({
            'ts': time.time(),
            'task_id': task_id,
            'kind': kind,
            'data': data,
        })

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + '.tmp')
        blob = {
            'run_id': self.run_id,
            'started_at': self.started_at,
            'saved_at': time.time(),
            'tasks': [t.to_dict() for t in self.tasks],
            'events': self.events,
        }
        tmp.write_text(json.dumps(blob, indent=2, default=str),
                       encoding='utf-8')
        tmp.replace(self.path)

    @classmethod
    def load(cls, path: Path) -> 'State':
        # Defer import to avoid circular dep at module load time.
        from .runners import rebuild_resolver_from_meta

        data = json.loads(Path(path).read_text(encoding='utf-8'))
        s = cls(Path(path), data.get('run_id'))
        s.started_at = data.get('started_at', time.time())
        s.events = data.get('events', [])
        for td in data.get('tasks', []):
            meta = dict(td.get('meta', {}))
            t = Task(
                id=td['id'],
                kind=TaskKind(td['kind']),
                cmd=list(td['cmd']),
                log=Path(td['log']),
                deps=list(td.get('deps', [])),
                timeout_min=td.get('timeout_min', 60),
                stall_min=td.get('stall_min', 20),
                gpu_min_free_gb=td.get('gpu_min_free_gb', 0.0),
                retries_left=td.get('retries_left', 2),
                meta=meta,
                status=TaskStatus(td.get('status', 'pending')),
                rc=td.get('rc'),
                started_at=td.get('started_at'),
                finished_at=td.get('finished_at'),
                # Rebuild callback from persisted meta so bench/tier can
                # re-resolve --cur-adapter / --peft on resume.
                resolve_cmd=rebuild_resolver_from_meta(meta),
            )
            # RUNNING state can't survive a scheduler crash — the
            # subprocess was orphaned. Reset to PENDING so it re-spawns.
            if t.status == TaskStatus.RUNNING:
                t.status = TaskStatus.PENDING
                t.started_at = None
            s.add_task(t)
        return s

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return {
            'run_id': self.run_id,
            'n_tasks': len(self.tasks),
            'status_counts': counts,
            'elapsed_min': (time.time() - self.started_at) / 60.0,
        }
