"""Jemma — offline brain-response Discord bot.

Topic-agnostic: processes any short recorded media (video, audio, image
sequence) across any subject domain. No inherent content filtering is
applied; outputs are for educational and research use only and do not
constitute medical advice.

Flow on every media attachment (or /jemma-demo):

1. React 🎬 on the user's message (acknowledged).
2. Post progress comment #1. Progressively edit it as each stage finishes.
3. Stage A (~2 s): Gemma vision — objective description of the clip.
4. Stage B (~10-20 s): TRIBE text-only on Gemma's description -> quick
   language-cortex narration.
5. Stage C (~4-7 min): TRIBE full multimodal (video+audio+text).
6. Multi-atlas BrainAnalysis (Harvard-Oxford + Jülich, ~30 s).
7. Post comment #2 — embed with TRIBE metrics + cortex PNG + narration tiers.

Demo mode (/jemma-demo):
  Generates all 7 expertise tiers (toddler → researcher) and posts them as a
  series of follow-up messages in the channel. The main embed has the cortex
  PNG and the general-audience narration; the remaining tiers are separate
  messages so each audience gets a clean, shareable block.

Pipeline queue: jobs are processed serially by a single worker task.
Failed jobs are retried up to MAX_RETRIES times before giving up.

Graceful shutdown: Ctrl-C (or SIGTERM on Unix) posts an offline notice to
DISCORD_STATUS_CHANNEL_ID, cancels the worker, then closes the client.
"""
from __future__ import annotations

import asyncio
import collections
import heapq
import io
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands

from . import analysis as _analysis
from . import config, media_gate, tiers
from .hwmon import _query_nvidia_smi
from .logger import log
from .model_manager import ModelTier, get_manager
from .pipeline import load_model, run_inference, run_inference_text_only
from .results_store import save_result, list_results
from .scheduler import start_scheduler, stop_scheduler
from .tunnel import get_public_url, viewer_url
from .visualize import render_peak_cortex

# ── GCP remote inference (optional) ──────────────────────────────────────────
# When GCP_INFERENCE=1 in .env, Stage C runs on a GCP preemptible L4 VM
# instead of the local GPU. The local GPU stays free for other work.
_GCP_INFERENCE = os.environ.get('GCP_INFERENCE', '0').strip() in ('1', 'true', 'yes')
if _GCP_INFERENCE:
    try:
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
        from gcp.remote_inference import run_remote_inference as _run_remote_inference
        log.info('[gcp] Remote inference mode ENABLED (GCP_INFERENCE=1)')
    except ImportError as _e:
        _GCP_INFERENCE = False
        log.warning('[gcp] GCP_INFERENCE=1 but gcp/remote_inference.py not importable: %s', _e)


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB  = config.MAX_UPLOAD_MB
ALLOWED_EXT    = {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac", ".m4a"}
MAX_RETRIES    = config.MAX_RETRIES
TRIBE_MAX_SECS = config.TRIBE_MAX_SECS

# ── Discord RBAC: role name sets (case-insensitive) ───────────────────────────
_STAFF_ROLES      = {r.strip().lower() for r in config.RBAC_STAFF_ROLES}
_RESEARCHER_ROLES = {r.strip().lower() for r in config.RBAC_RESEARCHER_ROLES}
_VERIFIED_ROLES   = {r.strip().lower() for r in config.RBAC_VERIFIED_ROLES}

# ── Rate limiter: user_id → deque of job submission timestamps ────────────────
_rate_tracker: dict[int, collections.deque] = collections.defaultdict(
    lambda: collections.deque()
)

# ── Priority queue (heap) — (priority, seq, _PipelineJob) ────────────────────
# Lower priority number = processed first.
# seq is a monotonically increasing counter used to break ties (FIFO within tier).
_pq_heap: list = []
_pq_seq: int = 0
_pq_event = asyncio.Event()

DISCLAIMER = (
    "Predictions from TRIBE v2 (CC-BY-NC 4.0, 25-subject group average) and "
    "Gemma 4 E4B. For educational and research use only. Not a diagnostic tool, "
    "not medical advice. Users are responsible for content-law compliance."
)

TIER_LABELS: dict[int, str] = {
    0: "🧒 Toddler / Age 3–5",
    1: "👵 General adult / no science background",
    2: "👥 Curious adult / general public",
    3: "📚 High school student",
    4: "🎓 College-educated adult",
    5: "🩺 Clinician / medical professional",
    6: "🔬 Neuroscience researcher / ML scientist",
}

REACT_ACK    = "\N{Clapper Board}"
REACT_VISION = "\N{Eyes}"
REACT_QUICK  = "\N{High Voltage Sign}"
REACT_BRAIN  = "\N{Brain}"
REACT_DONE   = "\N{White Heavy Check Mark}"
REACT_ERROR  = "\N{Cross Mark}"


# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class _PipelineJob:
    media_path:   Path
    display_name: str
    channel:      discord.abc.Messageable
    user_message: discord.Message | None
    progress_msg: discord.Message
    attempt:      int = 0
    is_demo:      bool = False
    priority:     int = config.QUEUE_PRIORITY_GUEST   # lower = higher priority
    user_id:      int = 0
    model_tier:   ModelTier = ModelTier.FAST
    user_roles:   frozenset = field(default_factory=frozenset)

    # Make comparable for heapq (compare only on priority)
    def __lt__(self, other: '_PipelineJob') -> bool:
        return self.priority < other.priority


_job_queue: asyncio.Queue[_PipelineJob] = asyncio.Queue()   # kept for back-compat


# ── RBAC helpers ─────────────────────────────────────────────────────────────

def _get_member_role_names(member: discord.Member | None) -> frozenset[str]:
    """Return lowercase role names for a guild member."""
    if member is None or not hasattr(member, 'roles'):
        return frozenset()
    return frozenset(r.name.lower() for r in member.roles if r.name != '@everyone')


def _classify_user(roles: frozenset[str]) -> tuple[int, ModelTier]:
    """
    Return (queue_priority, model_tier) for a member's role set.
    Higher role = lower priority number = gets processed sooner.
    """
    if roles & _STAFF_ROLES:
        return config.QUEUE_PRIORITY_STAFF, ModelTier.EXPERT
    if roles & _RESEARCHER_ROLES:
        return config.QUEUE_PRIORITY_RESEARCHER, ModelTier.EXPERT
    if roles & _VERIFIED_ROLES:
        return config.QUEUE_PRIORITY_VERIFIED, ModelTier.DEEP
    return config.QUEUE_PRIORITY_GUEST, ModelTier.FAST


def _check_rate_limit(user_id: int, priority: int) -> bool:
    """
    Return True if user is within their rate limit.
    Staff are never rate-limited.
    """
    if priority == config.QUEUE_PRIORITY_STAFF:
        return True

    limits = {
        config.QUEUE_PRIORITY_RESEARCHER: config.RATE_LIMIT_RESEARCHER_PER_HOUR,
        config.QUEUE_PRIORITY_VERIFIED:   config.RATE_LIMIT_VERIFIED_PER_HOUR,
        config.QUEUE_PRIORITY_GUEST:      config.RATE_LIMIT_GUEST_PER_HOUR,
    }
    max_per_hour = limits.get(priority, 1)

    now = time.time()
    dq  = _rate_tracker[user_id]

    # Remove timestamps older than 1 hour
    while dq and now - dq[0] > 3600:
        dq.popleft()

    return len(dq) < max_per_hour


def _record_job(user_id: int) -> None:
    """Record a job submission for rate limiting."""
    _rate_tracker[user_id].append(time.time())


def _enqueue_priority(job: '_PipelineJob') -> None:
    """Push job onto the priority heap and signal the worker."""
    global _pq_seq
    heapq.heappush(_pq_heap, (job.priority, _pq_seq, job))
    _pq_seq += 1
    _pq_event.set()


async def _dequeue_priority() -> '_PipelineJob':
    """Async-wait for a job on the priority heap."""
    while True:
        if _pq_heap:
            _, _, job = heapq.heappop(_pq_heap)
            if not _pq_heap:
                _pq_event.clear()
            return job
        _pq_event.clear()
        await _pq_event.wait()
_worker_task: asyncio.Task | None = None
_shutdown_event = asyncio.Event()


# ── Client subclass — posts offline status on close() ────────────────────────

class _JemmaClient(discord.Client):
    async def close(self) -> None:
        if _shutdown_event.is_set():
            await super().close()
            return
        _shutdown_event.set()
        log.info("bot closing; announcing offline")
        stop_scheduler()
        await _post_status("🔴 **Jemma is going offline.**")
        if _worker_task and not _worker_task.done():
            _worker_task.cancel()
            try:
                await asyncio.wait_for(_worker_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await super().close()
        log.info("bot closed")


intents = discord.Intents.default()
intents.message_content = True
client = _JemmaClient(intents=intents)
tree = app_commands.CommandTree(client)


# ── Small helpers ─────────────────────────────────────────────────────────────

async def _post_status(text: str) -> None:
    chan_id = config.DISCORD_STATUS_CHANNEL_ID
    if not chan_id:
        return
    try:
        channel = client.get_channel(int(chan_id))
        if channel:
            await channel.send(text)
    except Exception as exc:
        log.warning("status channel post failed", extra={"error": str(exc)})


async def _edit(progress_msg: discord.Message, state: "_ProgressState") -> None:
    try:
        await progress_msg.edit(content=state.render())
    except discord.HTTPException as exc:
        log.warning("progress edit failed", extra={"error": str(exc)})


async def _react(msg: discord.Message, emoji: str) -> None:
    try:
        await msg.add_reaction(emoji)
    except discord.HTTPException:
        pass


def _is_media(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT


def _truncate(text: str, limit: int) -> str:
    text = text.strip() or "(Gemma returned an empty response.)"
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


# ── Duration auto-trim ────────────────────────────────────────────────────────

async def _auto_trim(media_path: Path, max_secs: float = TRIBE_MAX_SECS) -> tuple[Path, str]:
    """Probe duration; if > max_secs, trim with ffmpeg and return (new_path, note)."""
    from .gemma_vision import _NOWWIN, _probe_duration
    import shutil as _sh

    dur = await asyncio.to_thread(_probe_duration, media_path)
    if dur <= max_secs + 2:  # 2 s tolerance
        return media_path, ""

    trimmed = config.UPLOAD_DIR / f"trimmed_{media_path.stem}_{max_secs:.0f}s{media_path.suffix}"
    ff = _sh.which("ffmpeg") or "ffmpeg"

    def _do() -> subprocess.CompletedProcess:
        return subprocess.run(
            [ff, "-y", "-i", str(media_path), "-t", str(max_secs),
             "-vf", "scale=480:trunc(ow/a/2)*2",
             "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-r", "24",
             "-c:a", "aac", "-b:a", "96k", "-ar", "16000",
             str(trimmed)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=_NOWWIN,
        )

    proc = await asyncio.to_thread(_do)
    if proc.returncode != 0 or not trimmed.exists():
        log.warning("auto-trim failed; using original", extra={"file": media_path.name,
                    "err": proc.stderr.decode(errors="replace")[-200:]})
        return media_path, ""

    log.info("auto-trimmed", extra={"orig_s": round(dur, 1), "new_s": max_secs,
                                    "file": media_path.name})
    return trimmed, f"_(clip trimmed to {max_secs:.0f}s — TRIBE v2 analyzes up to 50s)_"


# ── Pipeline worker ───────────────────────────────────────────────────────────

async def _worker() -> None:
    log.info("pipeline worker started")
    while not _shutdown_event.is_set():
        try:
            job = await asyncio.wait_for(_dequeue_priority(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        log.info("pipeline job dequeued", extra={
            "file":        job.display_name,
            "attempt":     job.attempt + 1,
            "demo":        job.is_demo,
            "priority":    job.priority,
            "model_tier":  job.model_tier.name,
            "queue_depth": len(_pq_heap),
        })

        try:
            await _run_pipeline(
                channel=job.channel,
                user_message=job.user_message,
                progress_msg=job.progress_msg,
                media_path=job.media_path,
                display_name=job.display_name,
                is_demo=job.is_demo,
                model_tier=job.model_tier,
                user_roles=job.user_roles,
            )
            log.info("pipeline job completed", extra={"file": job.display_name})
        except Exception as exc:
            next_attempt = job.attempt + 1
            log.error("pipeline job failed", exc_info=True, extra={
                "file": job.display_name, "attempt": next_attempt, "error": str(exc),
            })
            if next_attempt < MAX_RETRIES:
                job.attempt = next_attempt
                _enqueue_priority(job)           # re-queue with same priority
                log.info("pipeline job requeued", extra={
                    "file": job.display_name, "attempt": next_attempt + 1, "max": MAX_RETRIES,
                })
                try:
                    await job.progress_msg.edit(
                        content=(f"⚠️ Attempt {next_attempt} failed — retrying "
                                 f"({next_attempt}/{MAX_RETRIES})...\n`{exc}`")
                    )
                except discord.HTTPException:
                    pass
            else:
                log.error("pipeline job exhausted retries", extra={
                    "file": job.display_name, "max_retries": MAX_RETRIES,
                })
                try:
                    await job.progress_msg.edit(
                        content=f"❌ Pipeline failed after {MAX_RETRIES} attempts: `{exc}`"
                    )
                    await job.progress_msg.add_reaction(REACT_ERROR)
                except discord.HTTPException:
                    pass

    log.info("pipeline worker stopped")


# ── Discord events ────────────────────────────────────────────────────────────

@client.event
async def on_ready() -> None:
    log.info("logged in", extra={"user": str(client.user), "id": client.user.id})

    if config.DISCORD_GUILD_ID:
        guild = discord.Object(id=int(config.DISCORD_GUILD_ID))
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
        log.info("commands synced to guild", extra={"guild": config.DISCORD_GUILD_ID})
    else:
        await tree.sync()
        log.info("global commands synced (may take ~1 h)")

    log.info("pre-warming TRIBE v2 (expect ~6 s)...")
    await asyncio.to_thread(load_model)
    log.info("TRIBE v2 pre-warmed; Jemma is ready")

    # Warm the FAST model (E4B) — keep alive 60 min so the first job gets it instantly
    log.info("pre-warming Ollama FAST model (%s)...", config.OLLAMA_MODEL_FAST)
    asyncio.create_task(asyncio.to_thread(get_manager().warm_fast_model))

    async def _restart_worker() -> None:
        global _worker_task
        _worker_task = asyncio.create_task(_worker(), name="pipeline-worker")
        log.info("pipeline worker restarted by scheduler")

    global _worker_task
    _worker_task = asyncio.create_task(_worker(), name="pipeline-worker")

    start_scheduler(
        post_status_fn=_post_status,
        job_queue=_job_queue,           # kept for scheduler health checks (qsize=0 OK)
        worker_task_getter=lambda: _worker_task,
        restart_worker_fn=_restart_worker,
    )

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(client.close()),
        )

    await _post_status(
        f"🟢 **Jemma is online.** Logged in as `{client.user}`. "
        "TRIBE v2 pre-warmed. Drop any short media file for full brain-response analysis."
    )


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if not message.attachments:
        return
    if (
        config.DISCORD_ALLOWED_CHANNEL_ID
        and str(message.channel.id) != config.DISCORD_ALLOWED_CHANNEL_ID
    ):
        return

    attachment = next((a for a in message.attachments if _is_media(a.filename)), None)
    if attachment is None:
        return
    if attachment.size > MAX_UPLOAD_MB * 1024 * 1024:
        await message.reply(
            f"That clip is {attachment.size/1e6:.1f} MB — too large "
            f"(>{MAX_UPLOAD_MB} MB). Trim it and try again."
        )
        return

    # ── RBAC: classify user tier & priority ───────────────────────────────────
    member     = message.author if isinstance(message.author, discord.Member) else None
    user_roles = _get_member_role_names(member)
    priority, model_tier = _classify_user(user_roles)
    user_id    = message.author.id

    # ── Queue saturation guard ────────────────────────────────────────────────
    if len(_pq_heap) >= config.QUEUE_MAX_LENGTH:
        await message.reply(
            f"⏳ Queue full ({len(_pq_heap)}/{config.QUEUE_MAX_LENGTH}). "
            "Please try again in a few minutes."
        )
        return

    # ── Rate limit ────────────────────────────────────────────────────────────
    if not _check_rate_limit(user_id, priority):
        dq           = _rate_tracker[user_id]
        reset_in_s   = max(0, 3600 - (time.time() - dq[0]))
        limits_map   = {
            config.QUEUE_PRIORITY_RESEARCHER: config.RATE_LIMIT_RESEARCHER_PER_HOUR,
            config.QUEUE_PRIORITY_VERIFIED:   config.RATE_LIMIT_VERIFIED_PER_HOUR,
            config.QUEUE_PRIORITY_GUEST:      config.RATE_LIMIT_GUEST_PER_HOUR,
        }
        max_ph = limits_map.get(priority, 1)
        await message.reply(
            f"⏱️ You've reached your limit of **{max_ph} job(s)/hour**. "
            f"Resets in **{reset_in_s / 60:.0f} min**.\n"
            "_Ask a moderator for the **Verified** or **Researcher** role for more capacity._"
        )
        return

    _record_job(user_id)

    log.info("attachment received", extra={
        "file":       attachment.filename,
        "size_mb":    round(attachment.size / 1e6, 2),
        "channel":    str(message.channel),
        "author":     str(message.author),
        "priority":   priority,
        "model_tier": model_tier.name,
    })

    # ── Enterprise security validation ────────────────────────────────────────
    # Download attachment bytes then validate (magic bytes, ffprobe, entropy,
    # ClamAV, deduplication, path-traversal prevention, codec-bomb detection).
    try:
        raw_bytes = await attachment.read()
    except Exception as exc:
        await message.reply(
            f"❌ Failed to download your attachment: {exc}\n"
            "Please try re-uploading the file."
        )
        return

    from .file_validator import validate_and_save, ValidationError as FileValError
    try:
        validated = await validate_and_save(
            raw_bytes=raw_bytes,
            original_filename=attachment.filename,
            upload_dir=config.UPLOAD_DIR,
            max_bytes=MAX_UPLOAD_MB * 1024 * 1024,
        )
        dest = validated.path
        if validated.is_duplicate:
            try:
                await message.add_reaction("♻️")
            except discord.HTTPException:
                pass
    except FileValError as exc:
        # Map validation error codes to user-friendly emoji
        _err_icons = {
            "unsupported_type": "🚫",
            "too_large":        "📦",
            "too_small":        "❓",
            "bad_magic":        "🎭",
            "high_entropy":     "🔒",
            "corrupt_file":     "💔",
            "ffprobe_timeout":  "⏱️",
            "too_long":         "⏳",
            "too_many_streams": "💣",
            "bitrate_bomb":     "💣",
            "malware_detected": "🦠",
            "disk_full":        "💾",
            "path_traversal":   "🚨",
        }
        icon = _err_icons.get(exc.code, "❌")
        await message.reply(
            f"{icon} **File rejected:** {exc.reason}\n"
            "-# If you think this is an error, contact a moderator."
        )
        log.warning("file rejected user=%s code=%s reason=%s", user_id, exc.code, exc.reason)
        return

    try:
        await message.add_reaction(REACT_ACK)
    except discord.HTTPException:
        pass

    tier_badge = {
        config.QUEUE_PRIORITY_STAFF:      "⚡ **Staff** — Expert model (31B dense)",
        config.QUEUE_PRIORITY_RESEARCHER: "🔬 **Researcher** — Expert model (31B dense)",
        config.QUEUE_PRIORITY_VERIFIED:   "✅ **Verified** — Deep model (26B MoE)",
        config.QUEUE_PRIORITY_GUEST:      "👤 **Guest** — Standard analysis",
    }.get(priority, "")

    queue_pos = len(_pq_heap) + 1   # approximate position (job not yet pushed)
    queue_note = (f" · position ~{queue_pos} in queue" if queue_pos > 1 else "")

    progress_msg = await message.reply(
        f"\N{Clapper Board} Received **{attachment.filename}**.\n"
        f"{tier_badge}{queue_note}\n"
        "Initializing Gemma 4 and TRIBE v2 pipeline..."
    )

    job = _PipelineJob(
        media_path=dest,
        display_name=attachment.filename,
        channel=message.channel,
        user_message=message,
        progress_msg=progress_msg,
        is_demo=False,
        priority=priority,
        user_id=user_id,
        model_tier=model_tier,
        user_roles=user_roles,
    )
    _enqueue_priority(job)
    log.info("job enqueued", extra={
        "file":        attachment.filename,
        "priority":    priority,
        "model_tier":  model_tier.name,
        "queue_depth": len(_pq_heap),
    })


# ── Slash commands ────────────────────────────────────────────────────────────

@tree.command(name="jemma-demo",
              description="Full showcase: all 7 expertise tiers on the packaged demo clip.")
async def cmd_demo(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    if not config.DEMO_VIDEO.exists():
        await interaction.followup.send(
            "❌ Demo clip not found. Run `python -m bot.make_demo_asset` first."
        )
        return
    try:
        start_msg = await interaction.followup.send(
            "\N{Clapper Board} Running full demo — all 7 expertise tiers incoming...",
            wait=True,
        )
        job = _PipelineJob(
            media_path=config.DEMO_VIDEO,
            display_name=config.DEMO_VIDEO.name,
            channel=interaction.channel,
            user_message=None,
            progress_msg=start_msg,
            is_demo=True,
            priority=config.QUEUE_PRIORITY_STAFF,   # demo always high priority
            model_tier=ModelTier.EXPERT,
        )
        _enqueue_priority(job)
        log.info("demo job enqueued", extra={"file": config.DEMO_VIDEO.name})
    except Exception as exc:
        log.error("demo command failed", exc_info=True)
        await interaction.followup.send(f"Could not start demo: `{exc}`")


@tree.command(name="jemma-status", description="GPU, queue, and model health.")
async def cmd_status(interaction: discord.Interaction) -> None:
    gpu = _query_nvidia_smi()
    queue_depth  = len(_pq_heap)
    worker_alive = bool(_worker_task and not _worker_task.done())

    if gpu is None:
        gpu_line = "GPU: unavailable (nvidia-smi not found)"
    else:
        vram, util, temp, power = gpu
        gpu_line = (f"GPU VRAM: **{vram:.1f} GB** | util **{util}%** | "
                    f"temp **{temp}°C** | **{power:.0f} W**")

    mgr_status = get_manager().status_report()

    msg = (
        f"\N{Brain} **Jemma status**\n"
        f"{gpu_line}\n"
        f"Ollama: `{config.OLLAMA_URL}`\n"
        f"  FAST: `{config.OLLAMA_MODEL_FAST}` · "
        f"DEEP: `{config.OLLAMA_MODEL_DEEP}` · "
        f"EXPERT: `{config.OLLAMA_MODEL_EXPERT}`\n"
        f"{mgr_status}\n"
        f"TRIBE weights: `{config.WEIGHTS_DIR.name}` | "
        f"`duration_trs={config.TRIBE_CONFIG_UPDATE['data.duration_trs']}`\n"
        f"Priority queue: **{queue_depth}** pending | "
        f"worker: **{'running' if worker_alive else 'stopped'}**\n"
        f"All inference is local — no data leaves this machine."
    )
    await interaction.response.send_message(msg, ephemeral=True)


@tree.command(name="jemma-queue", description="Show current queue depth and estimated wait.")
async def cmd_queue(interaction: discord.Interaction) -> None:
    depth = len(_pq_heap)
    worker_alive = bool(_worker_task and not _worker_task.done())

    if depth == 0:
        msg = "✅ Queue is empty — next job starts immediately."
    else:
        # Rough estimate: ~6 min per job (median full multimodal)
        est_min = depth * 6
        msg = (
            f"⏳ **{depth}** job(s) in queue · estimated wait: **~{est_min} min**\n"
            f"Worker: **{'running' if worker_alive else 'stopped'}**\n"
            f"_Use `/jemma-demo` for a fast showcase using the built-in demo clip._"
        )
    await interaction.response.send_message(msg, ephemeral=True)


@tree.command(name="jemma-roles",
              description="Show available roles, model tiers, and rate limits.")
async def cmd_roles(interaction: discord.Interaction) -> None:
    msg = (
        "**🧠 Jemma — Role Tiers & Capabilities**\n\n"
        "**👤 Guest** (unverified)\n"
        f"  • {config.RATE_LIMIT_GUEST_PER_HOUR} job/hour · Queue priority 4\n"
        f"  • Model: `{config.OLLAMA_MODEL_FAST}` (E4B, fast)\n\n"
        "**✅ Verified** (phone-verified member)\n"
        f"  • {config.RATE_LIMIT_VERIFIED_PER_HOUR} jobs/hour · Queue priority 3\n"
        f"  • Model: `{config.OLLAMA_MODEL_DEEP}` (26B MoE, deep analysis)\n\n"
        "**🔬 Researcher / Scientist / Neuroscientist**\n"
        f"  • {config.RATE_LIMIT_RESEARCHER_PER_HOUR} jobs/hour · Queue priority 2\n"
        f"  • Model: `{config.OLLAMA_MODEL_EXPERT}` (31B dense, expert quality)\n"
        f"  • Full 7-tier narration in analysis thread\n\n"
        "**⚡ Staff / Admin / Moderator**\n"
        "  • Unlimited · Queue priority 1 (first in line)\n"
        f"  • Model: `{config.OLLAMA_MODEL_EXPERT}` (31B dense)\n\n"
        "_Contact a moderator to be assigned the Verified or Researcher role._"
    )
    await interaction.response.send_message(msg, ephemeral=True)


@tree.command(
    name="jemma-view",
    description="Get the 3D browser URL for the latest (or a specific) brain analysis.",
)
@app_commands.describe(job_id="Specific job ID (leave blank for latest)")
async def cmd_view(interaction: discord.Interaction, job_id: str = '') -> None:
    """Return the Three.js viewer URL for a result."""
    from .results_store import latest_result_id, list_results
    await interaction.response.defer(ephemeral=True)

    rid = job_id.strip() or latest_result_id()
    if not rid:
        await interaction.followup.send(
            "❌ No saved results yet. Run the bot on a media file first, "
            "or use `/jemma-demo` to generate one."
        )
        return

    url = viewer_url(rid)
    base = get_public_url()
    public_note = (
        f"Public URL (anyone can open this): **{url}**"
        if base else
        f"Local URL (only works on your machine): `{url}`\n"
        f"To make it public, run `cloudflared` and set `JEMMABRAIN_PUBLIC_URL` in `.env`."
    )
    await interaction.followup.send(
        f"\N{Brain} **3D cortical viewer** — job `{rid}`\n{public_note}",
        ephemeral=True,
    )


@tree.command(
    name="jemma-view-all",
    description="Post 3D viewer links for ALL saved brain analyses in this channel.",
)
async def cmd_view_all(interaction: discord.Interaction) -> None:
    """Re-post viewer links for every saved result — useful to share with collaborators."""
    await interaction.response.defer(thinking=True)

    results = list_results()
    if not results:
        await interaction.followup.send(
            "❌ No saved results found. Run the bot on some media files first."
        )
        return

    base = get_public_url()
    lines = []
    for r in results[:20]:   # cap at 20 to avoid rate limits
        rid  = r['job_id']
        url  = viewer_url(rid, base)
        title = r['stimulus_title'] or r['media_filename'] or rid
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(r['timestamp']).strftime('%b %d %H:%M')
        lines.append(f"**{title}** _{ts}_ — {r['n_trs']} TRs · [View 3D]({url})")

    header = (
        f"\N{Brain} **All saved brain analyses** ({len(results)} total)"
        + (f"\nBase URL: `{base}`" if base else "\n⚠️ No public URL — links are local only (`localhost:5173`)")
        + "\n\n"
    )
    # Discord message limit is 2000 chars — chunk if needed
    chunks, cur = [], header
    for line in lines:
        candidate = cur + line + '\n'
        if len(candidate) > 1900:
            chunks.append(cur)
            cur = line + '\n'
        else:
            cur = candidate
    if cur.strip():
        chunks.append(cur)

    first = True
    for chunk in chunks:
        if first:
            await interaction.followup.send(chunk)
            first = False
        else:
            await interaction.channel.send(chunk)

    if len(results) > 20:
        await interaction.channel.send(
            f"_(showing newest 20 of {len(results)} results — use `/jemma-view` with a job ID for older ones)_"
        )


# ── Progress state ────────────────────────────────────────────────────────────

@dataclass
class _ProgressState:
    header: str
    vision: str = ""
    modality: str = ""
    trim_note: str = ""
    quick: str = ""
    full_status: str = ""

    def render(self) -> str:
        parts = [self.header]
        if self.vision:
            parts.append(f"\n\N{Eyes} **Gemma vision:** {self.vision}")
        if self.modality:
            parts.append(f"\N{Speech Balloon} _Dominant modality: {self.modality}_")
        if self.trim_note:
            parts.append(f"⚠️ {self.trim_note}")
        if self.quick:
            parts.append(f"\n\N{High Voltage Sign} **Quick read (text-only TRIBE):**\n{self.quick}")
        if self.full_status:
            parts.append(f"\n\N{Brain} {self.full_status}")
        return "\n".join(parts)[:1990]


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def _run_pipeline(
    channel: discord.abc.Messageable,
    user_message: discord.Message | None,
    progress_msg: discord.Message,
    media_path: Path,
    display_name: str,
    is_demo: bool = False,
    model_tier: ModelTier = ModelTier.FAST,
    user_roles: frozenset = frozenset(),
) -> None:
    t_total = time.time()
    log.info("pipeline start", extra={"file": display_name, "demo": is_demo})
    state = _ProgressState(header=f"\N{Brain} **{display_name}** — analyzing...")

    # ── Stage A: Gemma vision description ─────────────────────────────────────
    t_a = time.time()
    log.debug("stage A: Gemma vision", extra={"file": display_name})
    cls = await asyncio.to_thread(media_gate.classify, media_path)
    state.header = f"\N{Brain} **{display_name}** — {cls.content_type} · {cls.subject}"
    state.vision = cls.short_description()
    state.modality = cls.modality
    await _edit(progress_msg, state)
    await _react(progress_msg, REACT_VISION)
    log.info("stage A complete", extra={
        "file": display_name, "content_type": cls.content_type,
        "modality": cls.modality, "secs": round(time.time() - t_a, 1),
    })

    # ── Duration check: auto-trim if > TRIBE_MAX_SECS ─────────────────────────
    media_path, trim_note = await _auto_trim(media_path)
    if trim_note:
        state.trim_note = trim_note
        await _edit(progress_msg, state)

    # ── Stage B: text-only TRIBE fast path ────────────────────────────────────
    t_b = time.time()
    log.debug("stage B: text-only TRIBE", extra={"file": display_name})
    quick_desc = cls.short_description()
    quick_result = await asyncio.to_thread(run_inference_text_only, quick_desc)
    quick_text = await asyncio.to_thread(tiers.narrate_quick, quick_result, quick_desc)
    quick_secs = time.time() - t_b
    state.quick = f"{quick_text}\n_(text-only pass, {quick_secs:.0f} s)_"
    state.full_status = (
        "Running **full multimodal** TRIBE v2 (V-JEPA2 + wav2vec-BERT + "
        "Llama-3.2-3B). Expect ~4-7 min. Jemma is thinking..."
    )
    await _edit(progress_msg, state)
    await _react(progress_msg, REACT_QUICK)
    await _react(progress_msg, REACT_BRAIN)
    log.info("stage B complete", extra={"file": display_name, "secs": round(quick_secs, 1)})

    # ── Stage C: full multimodal TRIBE (local or GCP remote) ─────────────────
    t_c = time.time()
    log.debug("stage C: full multimodal TRIBE", extra={"file": display_name})

    if _GCP_INFERENCE:
        # Off-load to preemptible GCP GPU VM — zero local GPU usage.
        state.full_status = (
            "Uploading to Google Cloud and launching GPU inference VM "
            "(NVIDIA L4, preemptible). Estimated 10-15 min..."
        )
        await _edit(progress_msg, state)

        async def _gcp_progress(msg: str):
            state.full_status = f"[GCP] {msg}"
            await _edit(progress_msg, state)

        gcp_job_id = f"job_{int(time.time())}_{media_path.stem[:20]}"
        completed_id = await _run_remote_inference(
            media_path, gcp_job_id, display_name, _gcp_progress
        )

        if completed_id is None:
            # Fall back to local inference on GCP failure
            log.warning("[gcp] Remote inference failed, falling back to local GPU")
            state.full_status = "GCP inference failed — falling back to local GPU..."
            await _edit(progress_msg, state)
            result = await asyncio.to_thread(run_inference, media_path)
        else:
            # Load result from the synced local file
            from .results_store import load_result_preds, load_result, RESULTS_DIR
            import json as _json, numpy as _np
            meta = _json.loads((RESULTS_DIR / f'{completed_id}_meta.json').read_text())
            preds = load_result_preds(completed_id)
            # Reconstruct a minimal result object compatible with downstream code
            import types as _types
            result = _types.SimpleNamespace(
                preds=preds,
                peak_t=float(meta.get('analysis', {}).get('temporal', {}).get('peak_s', 0)),
                media_path=media_path,
            )
    else:
        result = await asyncio.to_thread(run_inference, media_path)

    full_secs = time.time() - t_c
    log.info("stage C complete", extra={
        "file": display_name, "shape": list(result.preds.shape),
        "secs": round(full_secs, 1), "peak_t": result.peak_t,
    })

    # ── Multi-atlas BrainAnalysis ─────────────────────────────────────────────
    state.full_status = "Computing multi-atlas brain analysis (Harvard-Oxford, Jülich)..."
    await _edit(progress_msg, state)
    brain_analysis = await asyncio.to_thread(
        _analysis.analyse, result,
        harvard_oxford=True, juelich=True, brainnetome=False,
    )
    brain_ctx = brain_analysis.gemma_context()
    log.info("brain analysis complete", extra={
        "file": display_name, "secs": brain_analysis.analysis_seconds,
        "dominant_net": brain_analysis.dominant_network,
        "vertices_above_1sd": brain_analysis.vertices_above_1sd,
    })

    # ── Render peak cortex PNG ────────────────────────────────────────────────
    peak_png = await asyncio.to_thread(render_peak_cortex, result)
    label = f"{display_name} — {cls.short_description()}"

    # ── Tier narration — select tiers based on user model tier ────────────────
    if is_demo:
        state.full_status = "Generating all 7 expertise tiers (Gemma 4)..."
        await _edit(progress_msg, state)
        all_tiers: dict[int, str] = {}
        for tier_idx in range(7):
            tier_short = TIER_LABELS[tier_idx].split(" ", 1)[1]
            state.full_status = f"Tier {tier_idx + 1}/7: {tier_short}..."
            await _edit(progress_msg, state)
            all_tiers[tier_idx] = await asyncio.to_thread(
                tiers.narrate_tier, result, label, tier_idx, brain_ctx
            )
            log.debug(f"tier {tier_idx} narration done",
                      extra={"file": display_name, "chars": len(all_tiers[tier_idx])})
        narr = None
    else:
        # Expert users get deeper tiers; guests get the layperson + clinician
        if model_tier == ModelTier.EXPERT:
            state.full_status = "Generating expert narrations — 31B model (Gemma 4)..."
        elif model_tier == ModelTier.DEEP:
            state.full_status = "Generating deep narrations — 26B MoE (Gemma 4)..."
        else:
            state.full_status = "Generating narrations — E4B (Gemma 4)..."
        await _edit(progress_msg, state)
        narr = await asyncio.to_thread(tiers.narrate_tiered, result, label, brain_ctx)

    # ── Finalize progress comment ─────────────────────────────────────────────
    state.full_status = (
        f"Full analysis complete in **{full_secs/60:.1f} min**. "
        f"Peak at **t={result.peak_t/2:.1f}s**. See the reply below."
    )
    await _edit(progress_msg, state)
    await _react(progress_msg, REACT_DONE)

    # ── Build network summary for embed ──────────────────────────────────────
    nets_sorted = sorted(
        brain_analysis.network_means.items(), key=lambda kv: kv[1], reverse=True
    )
    net_lines = "\n".join(
        f"• **{_analysis._YEO7_FULL.get(k, k)}** {v:.3f} "
        f"({'L-dom' if brain_analysis.network_laterality.get(k, 0) > 0.1 else 'R-dom' if brain_analysis.network_laterality.get(k, 0) < -0.1 else 'bilateral'})"
        for k, v in nets_sorted[:5]
    )
    td = brain_analysis.temporal

    # ── Persist result + generate 3D viewer URL ───────────────────────────────
    job_id = f"job_{int(time.time())}_{Path(display_name).stem[:20].replace(' ', '_')}"
    narrations_dict: dict[int, str] = {}
    if is_demo:
        narrations_dict = all_tiers
    else:
        narrations_dict = {
            2: narr.layperson,
            5: narr.clinician,
            6: narr.researcher,
        }

    try:
        save_result(
            job_id=job_id,
            stimulus_title=display_name,
            preds=result.preds,
            analysis_dict=brain_analysis.to_dict(),
            narrations=narrations_dict,
            media_filename=display_name,
        )
        log.info("result saved", extra={"job_id": job_id})
    except Exception as e:
        log.warning("result save failed", extra={"error": str(e)})

    # Build 3D viewer URL (uses tunnel URL if available, else localhost)
    view_url = viewer_url(job_id)
    has_public_url = get_public_url() is not None

    # ── Post main embed ───────────────────────────────────────────────────────
    dom_net_name = _analysis._YEO7_FULL.get(brain_analysis.dominant_network, brain_analysis.dominant_network)
    embed = discord.Embed(
        title=f"\N{Brain} Predicted brain response — {display_name}",
        description=(
            f"**TRIBE v2** · {result.preds.shape[0]} TRs × {result.preds.shape[1]:,} vertices "
            f"(fsaverage5, 2 Hz)\n"
            f"Peak **t={td.get('peak_s', 0):.1f}s** · |z|={td.get('peak_z', 0):.3f} · "
            f"Rise: {td.get('rise_s', 0):.1f}s · "
            f"Above half-max: {td.get('duration_above_half_max_s', 0):.1f}s\n"
            f"Activated: **{brain_analysis.vertices_above_1sd:,}** / 20,484 vertices "
            f"above 1σ ({brain_analysis.activation_fraction_1sd * 100:.1f}% of cortex)\n"
            f"Dominant: **{dom_net_name}**\n\n"
            f"Top Yeo-7 networks:\n{net_lines}"
        ),
        colour=discord.Colour.from_rgb(88, 101, 242),
        url=view_url if has_public_url else None,
    )

    if is_demo:
        embed.add_field(
            name=f"{TIER_LABELS[2]}",
            value=_truncate(all_tiers[2], 1020),
            inline=False,
        )
    else:
        embed.add_field(
            name=f"{TIER_LABELS[2]}",
            value=_truncate(narr.layperson, 1020),
            inline=False,
        )
        embed.add_field(
            name=f"{TIER_LABELS[5]}",
            value=_truncate(narr.clinician, 1020),
            inline=False,
        )
        embed.add_field(
            name=f"{TIER_LABELS[6]}",
            value=_truncate(narr.researcher, 1020),
            inline=False,
        )

    embed.set_footer(text=DISCLAIMER)

    with peak_png.open("rb") as f:
        file = discord.File(io.BytesIO(f.read()), filename=peak_png.name)
    embed.set_image(url=f"attachment://{peak_png.name}")

    result_msg = await progress_msg.reply(embed=embed, file=file)
    total_secs = time.time() - t_total
    log.info("main embed posted", extra={
        "file": display_name, "total_secs": round(total_secs, 1),
        "reply_id": result_msg.id, "job_id": job_id,
    })

    # ── Post 3D viewer link ───────────────────────────────────────────────────
    await _post_viewer_link(result_msg, job_id, view_url, has_public_url)

    # ── Demo mode: post the remaining 6 tiers as a series of follow-up messages
    if is_demo:
        await _post_demo_tiers(result_msg, all_tiers, display_name)

    # ── Open analysis thread with full narrations + brain breakdown ───────────
    asyncio.create_task(
        _post_analysis_thread(result_msg, narrations_dict, brain_analysis, display_name, job_id)
    )

    # ── Push to central results feed ──────────────────────────────────────────
    asyncio.create_task(
        _post_results_feed(result, brain_analysis, narrations_dict, display_name, job_id,
                           view_url, has_public_url)
    )

    # ── Push to Three.js WebSocket (if server is running) ────────────────────
    asyncio.create_task(_push_to_webapp(result, brain_analysis, narrations_dict, display_name))

    log.info("pipeline complete", extra={
        "file": display_name, "total_secs": round(time.time() - t_total, 1),
    })


# ── 3D viewer link helper ─────────────────────────────────────────────────────

async def _post_viewer_link(
    anchor: discord.Message,
    job_id: str,
    view_url: str,
    has_public_url: bool,
) -> None:
    """Post a follow-up message with the 3D viewer URL."""
    if has_public_url:
        msg = (
            f"🌐 **View in 3D browser** → {view_url}\n"
            f"_(rotating cortical surface, 20,484 vertices, 5-second segments, "
            f"Yeo-7 networks, live playback)_"
        )
    else:
        msg = (
            f"🖥 **View locally** → `{view_url}`\n"
            f"_(open this URL in your browser while `python start_viz.py` is running — "
            f"or set `JEMMABRAIN_PUBLIC_URL` + cloudflared for a public link)_"
        )
    try:
        await anchor.reply(msg)
    except discord.HTTPException as e:
        log.warning("viewer link post failed", extra={"error": str(e)})


# ── WebSocket push (fire-and-forget) ─────────────────────────────────────────

async def _push_to_webapp(result, brain_analysis, narrations: dict, stimulus_title: str) -> None:
    """Push completed result to the Three.js webapp via the server's push_result API."""
    try:
        import importlib, sys as _sys
        _sys.path.insert(0, str(config.PROJECT_ROOT / 'webapp'))
        import server as _server
        await _server.push_result(result, brain_analysis, narrations, stimulus_title)
        log.info("pushed result to webapp WebSocket")
    except Exception as e:
        log.debug("webapp push skipped (server not running?): %s", e)


# ── Analysis thread (opened on the result embed) ─────────────────────────────

async def _post_analysis_thread(
    anchor: discord.Message,
    narrations: dict[int, str],
    brain_analysis,
    display_name: str,
    job_id: str,
) -> None:
    """
    Open a Discord thread on the result embed and post:
      • All available tier narrations (beyond what the embed shows)
      • Top ROIs with mean |z|
      • Temporal dynamics details
      • Data provenance note
    """
    try:
        thread = await anchor.create_thread(
            name=f"🧠 {display_name[:80]}",
            auto_archive_duration=1440,   # 24 h
        )
    except discord.HTTPException as e:
        log.warning("thread creation failed: %s", e)
        return

    # ── Post full narrations not shown in the embed ───────────────────────────
    tier_order = sorted(narrations.keys())
    narr_chunks: list[str] = []
    for t in tier_order:
        label_line = f"**{TIER_LABELS.get(t, f'Tier {t}')}**"
        body       = _truncate(narrations[t], 1800)
        narr_chunks.append(f"{label_line}\n{body}")

    # Post in groups to stay under Discord's 2000-char limit
    buf = ""
    for chunk in narr_chunks:
        candidate = buf + "\n\n" + chunk if buf else chunk
        if len(candidate) > 1900:
            await thread.send(buf)
            buf = chunk
        else:
            buf = candidate
    if buf:
        await thread.send(buf)

    # ── Brain analysis details ────────────────────────────────────────────────
    from . import analysis as _an
    td   = brain_analysis.temporal
    nets = sorted(brain_analysis.network_means.items(), key=lambda kv: kv[1], reverse=True)

    net_text = "\n".join(
        f"  • **{_an._YEO7_FULL.get(k, k)}** — mean |z| = {v:.3f}"
        for k, v in nets[:7]
    )
    detail_text = (
        f"**Brain network breakdown — `{job_id}`**\n"
        f"Peak activation: **t = {td.get('peak_s', 0):.1f}s** "
        f"(|z| = {td.get('peak_z', 0):.3f})\n"
        f"Rise time: {td.get('rise_s', 0):.1f}s · "
        f"Above half-max: {td.get('duration_above_half_max_s', 0):.1f}s\n"
        f"Activated vertices: **{brain_analysis.vertices_above_1sd:,}** / 20,484 "
        f"({brain_analysis.activation_fraction_1sd * 100:.1f}% of cortex)\n\n"
        f"**Yeo-7 network means (|z|):**\n{net_text}\n\n"
        f"_TRIBE v2 CC-BY-NC 4.0 · fsaverage5 · 2 Hz · Not medical advice._"
    )
    await thread.send(detail_text[:1990])
    log.info("analysis thread posted", extra={"job_id": job_id, "thread": thread.id})


# ── Central results feed ──────────────────────────────────────────────────────

async def _post_results_feed(
    result,
    brain_analysis,
    narrations: dict[int, str],
    display_name: str,
    job_id: str,
    view_url: str,
    has_public_url: bool,
) -> None:
    """Post a brief summary embed to the central #results-feed channel."""
    if not config.ENABLE_RESULTS_FEED:
        return
    chan_id = config.DISCORD_RESULTS_CHANNEL_ID
    if not chan_id:
        return
    try:
        feed_channel = client.get_channel(int(chan_id))
        if feed_channel is None:
            return

        from . import analysis as _an
        dom = brain_analysis.dominant_network
        dom_name = _an._YEO7_FULL.get(dom, dom)
        # Get the layperson narration for the feed preview
        preview_tier = min(narrations.keys(), key=lambda t: abs(t - 2))
        preview = _truncate(narrations.get(preview_tier, ""), 300)

        embed = discord.Embed(
            title=f"🧠 {display_name}",
            description=(
                f"Dominant network: **{dom_name}**\n"
                f"Activated: {brain_analysis.vertices_above_1sd:,} vertices "
                f"({brain_analysis.activation_fraction_1sd * 100:.1f}% cortex)\n\n"
                f"{preview}"
            ),
            colour=discord.Colour.from_rgb(88, 101, 242),
            url=view_url if has_public_url else None,
        )
        embed.set_footer(text=f"Job {job_id} · {result.preds.shape[0]} TRs · TRIBE v2")
        await feed_channel.send(embed=embed)
        log.info("results feed posted", extra={"job_id": job_id, "channel": chan_id})
    except Exception as exc:
        log.warning("results feed post failed: %s", exc)


# ── Demo tier follow-ups ──────────────────────────────────────────────────────

async def _post_demo_tiers(
    anchor: discord.Message,
    all_tiers: dict[int, str],
    display_name: str,
) -> None:
    """Post 6 follow-up messages with the remaining tiers (0,1,3,4 as text; 5,6 as embed)."""

    # Message 1: Tiers 0 + 1 (simple audiences)
    m1_text = (
        f"**{TIER_LABELS[0]}**\n{_truncate(all_tiers[0], 900)}"
        f"\n\n**{TIER_LABELS[1]}**\n{_truncate(all_tiers[1], 900)}"
    )
    m1 = await anchor.reply(m1_text)
    log.debug("demo tiers 0+1 posted", extra={"file": display_name})

    # Message 2: Tiers 3 + 4 (student / college)
    m2_text = (
        f"**{TIER_LABELS[3]}**\n{_truncate(all_tiers[3], 900)}"
        f"\n\n**{TIER_LABELS[4]}**\n{_truncate(all_tiers[4], 900)}"
    )
    m2 = await m1.reply(m2_text)
    log.debug("demo tiers 3+4 posted", extra={"file": display_name})

    # Message 3: Tiers 5 + 6 in an embed (longer technical content)
    embed56 = discord.Embed(
        title="Expert narrations",
        colour=discord.Colour.from_rgb(235, 69, 158),
    )
    embed56.add_field(
        name=TIER_LABELS[5],
        value=_truncate(all_tiers[5], 1020),
        inline=False,
    )
    embed56.add_field(
        name=TIER_LABELS[6],
        value=_truncate(all_tiers[6], 1020),
        inline=False,
    )
    embed56.set_footer(text=DISCLAIMER)
    await m2.reply(embed=embed56)
    log.info("demo all 7 tiers posted", extra={"file": display_name})


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not config.DISCORD_TOKEN:
        raise SystemExit(
            "DISCORD_TOKEN env var is required. See bot/README.md for setup."
        )
    log.info("Jemma starting up")
    client.run(config.DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
