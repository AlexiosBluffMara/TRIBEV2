"""APScheduler async health monitoring for Jemma.

Scheduled tasks
───────────────
  gpu-check      every 60 s   — warns to status channel if GPU temp > GPU_TEMP_WARN
                               or VRAM > GPU_VRAM_WARN
  worker-watch   every  5 min — restarts the pipeline worker if it died silently
  queue-stats    every 30 min — posts a one-line digest to the status channel

All tasks are coroutines; they are scheduled on the running asyncio event loop
via APScheduler's AsyncIOScheduler.

Usage (called from bot.py on_ready):
    from .scheduler import start_scheduler, stop_scheduler
    start_scheduler(post_status_fn, job_queue, worker_task_getter, restart_worker_fn)

The scheduler auto-stops on bot shutdown through _JemmaClient.close().
"""
from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .hwmon import _query_nvidia_smi
from .logger import log

GPU_TEMP_WARN  = 85    # °C — warn when sustained above this
GPU_VRAM_WARN  = 30.0  # GB — warn when VRAM use exceeds this

_scheduler: AsyncIOScheduler | None = None


def start_scheduler(
    post_status_fn,
    job_queue,
    worker_task_getter,
    restart_worker_fn,
) -> AsyncIOScheduler:
    """Start the health-monitoring scheduler. Call once from on_ready.

    Args:
        post_status_fn:    async (text: str) -> None — posts to status channel
        job_queue:         asyncio.Queue — the bot's pipeline queue
        worker_task_getter: () -> asyncio.Task | None — returns current worker task
        restart_worker_fn: async () -> None — called to relaunch a dead worker
    """
    global _scheduler

    if _scheduler and _scheduler.running:
        log.warning("scheduler already running; skipping start")
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="UTC")

    # ── GPU temperature / VRAM watch ─────────────────────────────────────────
    @_scheduler.scheduled_job("interval", seconds=60, id="gpu-check",
                               max_instances=1, coalesce=True)
    async def _gpu_check() -> None:
        gpu = _query_nvidia_smi()
        if gpu is None:
            return
        vram, util, temp, power = gpu
        if temp > GPU_TEMP_WARN or vram > GPU_VRAM_WARN:
            await post_status_fn(
                f"🌡️ **GPU alert** — {temp}°C · VRAM {vram:.1f} GB · "
                f"util {util}% · {power:.0f} W"
            )
            log.warning("GPU alert threshold exceeded",
                        extra={"temp": temp, "vram": vram, "util": util})

    # ── Pipeline worker liveness check ────────────────────────────────────────
    @_scheduler.scheduled_job("interval", minutes=5, id="worker-watch",
                               max_instances=1, coalesce=True)
    async def _worker_watch() -> None:
        wt = worker_task_getter()
        if wt is None:
            return
        if wt.done():
            exc = wt.exception()
            log.error("pipeline worker died; restarting",
                      extra={"exc": str(exc) if exc else "cancelled"})
            await post_status_fn(
                f"⚠️ **Pipeline worker died** (`{exc}`). Restarting..."
            )
            await restart_worker_fn()
            await post_status_fn("✅ Pipeline worker restarted.")

    # ── Periodic queue + GPU summary ──────────────────────────────────────────
    @_scheduler.scheduled_job("interval", minutes=30, id="queue-stats",
                               max_instances=1, coalesce=True)
    async def _queue_stats() -> None:
        depth = job_queue.qsize()
        gpu = _query_nvidia_smi()
        gpu_part = ""
        if gpu:
            vram, util, temp, power = gpu
            gpu_part = f" | GPU {temp}°C · {vram:.1f} GB · {power:.0f} W"
        log.info("periodic queue stats",
                 extra={"queue_depth": depth})
        await post_status_fn(
            f"📊 **Queue digest** — {depth} job(s) pending{gpu_part}"
        )

    _scheduler.start()
    log.info("scheduler started", extra={"jobs": len(_scheduler.get_jobs())})
    return _scheduler


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler (called from bot.close())."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("scheduler stopped")
    _scheduler = None
