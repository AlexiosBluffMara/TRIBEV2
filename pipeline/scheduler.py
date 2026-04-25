"""Priority scheduler with serial GPU worker + parallel CPU workers.

Design goals:

1. GPU is never idle between GPU tasks — as soon as one finishes, the
   next-highest-priority GPU task dispatches. CPU tasks run in parallel
   (up to `max_cpu`) so curriculum builds and compile steps happen
   during GPU windows.

2. Heaviest GPU tasks (FULL_TRAIN, 120 min) dispatch first so budget is
   spent on the tasks that actually need it before anything quick.

3. Every running process has a hard timeout AND a stall timeout: if the
   log file hasn't grown for `stall_min` minutes, kill early. Catches
   frozen HF downloads without eating the whole timeout window.

4. Self-heal: OOM → fallback_factory (e.g. halve LoRA rank) + retry.
   HF download error → bump stall_min and retry. Generic rc != 0 with
   retries_left → requeue. Hard timeout → fail (we already gave up).

5. All transitions persist to State immediately so a scheduler crash
   can be resumed cleanly.

6. Writes a heartbeat JSON every loop tick so the external dashboard
   can surface current state without parsing logs.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable, IO

from .gpu import (
    CREATE_NO_WINDOW, detect_error_signatures, gpu_free_gb, gpu_snapshot,
    wait_for_gpu,
)
from .state import State
from .tasks import Task, TaskStatus


class RunningTask:
    """A spawned subprocess with its associated log handle + deadlines."""

    def __init__(self, task: Task, proc: subprocess.Popen, log_fh: IO):
        self.task = task
        self.proc = proc
        self.log_fh = log_fh
        self.started_at = time.time()
        self.deadline = self.started_at + task.timeout_min * 60
        self.last_log_size = 0
        self.last_growth_t = self.started_at

    def poll(self) -> tuple[bool, int | None, str]:
        """Returns (is_done, rc, reason). reason ∈ {'ok','timeout','stall',''}."""
        rc = self.proc.poll()
        if rc is not None:
            return True, rc, 'ok'
        now = time.time()
        if now >= self.deadline:
            self._kill('# ABORTED: hard timeout')
            return True, -1, 'timeout'
        try:
            sz = self.task.log.stat().st_size
        except OSError:
            sz = self.last_log_size
        if sz > self.last_log_size:
            self.last_log_size = sz
            self.last_growth_t = now
        elif now - self.last_growth_t >= self.task.stall_min * 60:
            self._kill(f'# ABORTED: stalled ({self.task.stall_min} min '
                       'without log growth)')
            return True, -2, 'stall'
        return False, None, ''

    def _kill(self, note: str) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.log_fh.write(f'\n\n{note}\n')
            self.log_fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.log_fh.close()
        except Exception:
            pass


class Scheduler:
    def __init__(self, state: State, max_cpu: int = 2,
                 log: Callable[[str], None] = print,
                 heartbeat_path: Path | None = None,
                 poll_s: int = 5,
                 max_gpu_concurrent: int = 3,
                 gpu_safety_margin_gb: float = 1.5,
                 spawn_settle_s: float = 4.0):
        self.state = state
        self.max_cpu = max_cpu
        self.log = log
        self.heartbeat_path = heartbeat_path
        self.poll_s = poll_s
        # Hard cap on concurrent GPU subprocesses. Combined with the
        # live-free-VRAM admission check below, this lets us pack 2-3 bench
        # jobs side-by-side when VRAM allows while still serializing big
        # training jobs (which each need 14-22 GB free on their own).
        self.max_gpu_concurrent = max_gpu_concurrent
        # Safety margin: require `task.gpu_min_free_gb + margin` free before
        # admitting, so the incoming process has headroom to load weights
        # before another candidate starts measuring.
        self.gpu_safety_margin_gb = gpu_safety_margin_gb
        # After spawning a GPU task, wait this long before considering
        # another admission — gives the new process time to claim VRAM so
        # the next admission check sees the post-load free_gb.
        self.spawn_settle_s = spawn_settle_s
        self._gpu_running: list[RunningTask] = []
        self._cpu_running: list[RunningTask] = []
        self._last_gpu_spawn_t = 0.0
        self._start_wall = time.time()

    # -------- selection --------

    def _deps_met(self, task: Task) -> bool:
        for dep_id in task.deps:
            dep = self.state.get(dep_id)
            if dep is None:
                return False
            if dep.status != TaskStatus.DONE:
                return False
        return True

    def _pending_dispatchable(self) -> list[Task]:
        # Preserve insertion order within each (is_gpu, priority) bucket
        # so users who list hypotheses in preferred order see that order
        # honored. Python's list.sort is stable, so (gpu_first, priority)
        # keys are enough.
        out = [t for t in self.state.tasks
               if t.status == TaskStatus.PENDING and self._deps_met(t)]
        out.sort(key=lambda t: (not t.is_gpu(), -t.priority()))
        return out

    # -------- spawn --------

    def _spawn(self, task: Task) -> RunningTask | None:
        # Let the task rewrite its cmd now that deps are DONE.
        if task.resolve_cmd:
            try:
                new_cmd = task.resolve_cmd(task, self.state)
            except Exception as e:
                self.log(f'[sched] {task.id}: resolve_cmd raised {e!r}')
                new_cmd = None
            if new_cmd is None:
                task.status = TaskStatus.FAILED
                self.state.add_event(task.id, 'resolve_fail', {})
                self.state.save()
                return None
            task.cmd = new_cmd

        try:
            task.log.parent.mkdir(parents=True, exist_ok=True)
            fh = task.log.open('w', encoding='utf-8', errors='replace')
            fh.write(f'# cmd: {" ".join(shlex.quote(c) for c in task.cmd)}\n')
            fh.write(f'# kind: {task.kind.value}\n')
            fh.write(f'# started: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            fh.flush()
        except OSError as e:
            self.log(f'[sched] {task.id}: log open failed: {e}')
            task.status = TaskStatus.FAILED
            self.state.add_event(task.id, 'log_open_fail', {'err': str(e)})
            self.state.save()
            return None

        try:
            p = subprocess.Popen(
                task.cmd, stdout=fh, stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as e:
            self.log(f'[sched] {task.id}: popen failed: {e}')
            fh.close()
            task.status = TaskStatus.FAILED
            self.state.add_event(task.id, 'popen_fail', {'err': str(e)})
            self.state.save()
            return None

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.log(f'[sched] {task.id}: spawn pid={p.pid} '
                 f'(kind={task.kind.value}, timeout={task.timeout_min}m, '
                 f'stall={task.stall_min}m)')
        self.state.add_event(task.id, 'spawn',
                             {'pid': p.pid, 'cmd': list(task.cmd)})
        self.state.save()
        return RunningTask(task, p, fh)

    # -------- completion + healing --------

    def _handle_done(self, rt: RunningTask, rc: int, reason: str) -> None:
        task = rt.task
        task.finished_at = time.time()
        task.rc = rc
        rt.close()
        elapsed_min = (task.finished_at - (task.started_at or
                                           task.finished_at)) / 60.0

        if rc == 0 and reason == 'ok':
            task.status = TaskStatus.DONE
            self.log(f'[sched] {task.id}: DONE in {elapsed_min:.1f}m')
            self.state.add_event(task.id, 'done',
                                 {'elapsed_min': elapsed_min})
            self.state.save()
            return

        action = self._decide_heal(task, rc, reason)
        if action == 'retry':
            task.retries_left -= 1
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.finished_at = None
            task.rc = None
            self.log(f'[sched] {task.id}: {reason}/rc={rc} — RETRYING '
                     f'({task.retries_left} left, stall={task.stall_min}m)')
            self.state.add_event(task.id, 'retry',
                                 {'rc': rc, 'reason': reason,
                                  'retries_left': task.retries_left})
        else:
            task.status = TaskStatus.FAILED
            self.log(f'[sched] {task.id}: {reason}/rc={rc} — FAILED '
                     f'(after {elapsed_min:.1f}m)')
            self.state.add_event(task.id, 'failed',
                                 {'rc': rc, 'reason': reason,
                                  'elapsed_min': elapsed_min})
        self.state.save()

    def _decide_heal(self, task: Task, rc: int, reason: str) -> str:
        """Return 'retry' or 'fail'."""
        if task.retries_left <= 0:
            return 'fail'
        sigs = detect_error_signatures(task.log)
        if sigs['oom'] and task.fallback_factory:
            try:
                degraded = task.fallback_factory(task)
            except Exception as e:
                self.log(f'[sched] {task.id}: fallback raised {e!r}')
                return 'fail'
            # Rewrite in place so deps still point here.
            task.cmd = degraded.cmd
            task.timeout_min = degraded.timeout_min
            task.stall_min = degraded.stall_min
            task.meta = {**task.meta, **degraded.meta,
                         'fallback_applied': True}
            self.state.add_event(task.id, 'oom_fallback',
                                 {'new_cmd': list(task.cmd)})
            return 'retry'
        if sigs['oom']:
            return 'fail'  # no fallback available
        if sigs['hf_download']:
            task.stall_min = max(task.stall_min + 10,
                                 int(task.stall_min * 1.5))
            self.state.add_event(task.id, 'hf_retry',
                                 {'new_stall_min': task.stall_min})
            return 'retry'
        if sigs['transient']:
            return 'retry'
        if reason == 'stall':
            task.stall_min = int(task.stall_min * 1.5)
            self.state.add_event(task.id, 'stall_retry',
                                 {'new_stall_min': task.stall_min})
            return 'retry'
        if reason == 'timeout':
            # We already gave it the full window; retrying rarely helps.
            return 'fail'
        # Generic rc != 0
        return 'retry'

    # -------- heartbeat --------

    def _write_heartbeat(self) -> None:
        if not self.heartbeat_path:
            return
        try:
            blob = {
                'run_id': self.state.run_id,
                'ts': time.time(),
                'elapsed_min': (time.time() - self._start_wall) / 60.0,
                'gpu': gpu_snapshot(),
                'gpu_running': [rt.task.id for rt in self._gpu_running],
                'gpu_concurrent': len(self._gpu_running),
                'gpu_cap': self.max_gpu_concurrent,
                'cpu_running': [rt.task.id for rt in self._cpu_running],
                'counts': self.state.summary()['status_counts'],
                'tasks': [{
                    'id': t.id, 'kind': t.kind.value, 'status': t.status.value,
                    'rc': t.rc,
                    'elapsed_min': (
                        ((t.finished_at or time.time()) - t.started_at) / 60.0
                        if t.started_at else None),
                } for t in self.state.tasks],
            }
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.heartbeat_path.with_suffix(
                self.heartbeat_path.suffix + '.tmp')
            tmp.write_text(json.dumps(blob, indent=2, default=str),
                           encoding='utf-8')
            tmp.replace(self.heartbeat_path)
        except OSError:
            pass

    # -------- main loop --------

    def _can_admit_gpu(self, task: Task) -> tuple[bool, str]:
        """Return (admit, reason_if_not). Checks hard cap, settle window,
        and live free VRAM headroom."""
        if len(self._gpu_running) >= self.max_gpu_concurrent:
            return False, f'cap {self.max_gpu_concurrent} reached'
        # After spawning something, let VRAM settle before admitting another.
        if self._gpu_running and (
                time.time() - self._last_gpu_spawn_t < self.spawn_settle_s):
            return False, 'spawn settle'
        need = task.gpu_min_free_gb
        if need <= 0:
            return True, ''
        free = gpu_free_gb()
        budget = need + self.gpu_safety_margin_gb
        if free < budget:
            return False, f'free={free:.1f}GB < need={budget:.1f}GB'
        return True, ''

    def run(self) -> None:
        self.log(f'[sched] run {self.state.run_id}  '
                 f'{len(self.state.tasks)} tasks  max_cpu={self.max_cpu} '
                 f'max_gpu={self.max_gpu_concurrent} '
                 f'margin={self.gpu_safety_margin_gb:.1f}GB')
        while True:
            # 1. Poll + handle completions
            for rt in list(self._cpu_running):
                done, rc, reason = rt.poll()
                if done:
                    self._handle_done(rt, rc, reason)
                    self._cpu_running.remove(rt)
            for rt in list(self._gpu_running):
                done, rc, reason = rt.poll()
                if done:
                    self._handle_done(rt, rc, reason)
                    self._gpu_running.remove(rt)

            # 2. Dispatch new tasks. GPU tasks pack in free-VRAM order with
            #    a hard cap + safety margin; CPU tasks parallel up to max_cpu.
            pending = self._pending_dispatchable()
            deferred_gpu_reasons: dict[str, str] = {}
            for task in pending:
                if task.is_gpu():
                    admit, why = self._can_admit_gpu(task)
                    if not admit:
                        deferred_gpu_reasons[task.id] = why
                        # If no GPU tasks are running AND we have retries of
                        # waiting for headroom, do a bounded wait_for_gpu so
                        # we don't spin idle while the GPU drains from other
                        # consumers.
                        if (not self._gpu_running
                                and task.gpu_min_free_gb > 0
                                and 'free=' in why):
                            ok = wait_for_gpu(
                                task.gpu_min_free_gb
                                + self.gpu_safety_margin_gb,
                                max_wait_min=30, log=self.log)
                            if not ok:
                                task.status = TaskStatus.FAILED
                                self.log(f'[sched] {task.id}: '
                                         'GPU wait timeout')
                                self.state.add_event(
                                    task.id, 'gpu_wait_timeout', {})
                                self.state.save()
                                continue
                            # Re-check after wait.
                            admit, why = self._can_admit_gpu(task)
                            if not admit:
                                continue
                        else:
                            continue
                    rt = self._spawn(task)
                    if rt:
                        self._gpu_running.append(rt)
                        self._last_gpu_spawn_t = time.time()
                else:
                    if len(self._cpu_running) >= self.max_cpu:
                        continue
                    rt = self._spawn(task)
                    if rt:
                        self._cpu_running.append(rt)

            if deferred_gpu_reasons:
                # Log once per tick at debug level so it's not spammy.
                pass

            self._write_heartbeat()

            # 3. Exit if nothing left to do
            if (not self._gpu_running and not self._cpu_running
                    and not any(t.status == TaskStatus.PENDING
                                for t in self.state.tasks)):
                break

            time.sleep(self.poll_s)

        self.log(f'[sched] run complete: {self.state.summary()}')
