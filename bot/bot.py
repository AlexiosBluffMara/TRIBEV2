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
import io
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands

from . import analysis as _analysis
from . import config, media_gate, tiers
from .hwmon import _query_nvidia_smi
from .logger import log
from .pipeline import load_model, run_inference, run_inference_text_only
from .visualize import render_peak_cortex


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = 25
ALLOWED_EXT   = {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac"}
MAX_RETRIES   = 3
TRIBE_MAX_SECS = 50  # duration_trs=100 TRs × 2 Hz — hard limit from training

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
    media_path: Path
    display_name: str
    channel: discord.abc.Messageable
    user_message: discord.Message | None
    progress_msg: discord.Message
    attempt: int = 0
    is_demo: bool = False


_job_queue: asyncio.Queue[_PipelineJob] = asyncio.Queue()
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
            job = await asyncio.wait_for(_job_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        log.info("pipeline job dequeued", extra={
            "file": job.display_name,
            "attempt": job.attempt + 1,
            "demo": job.is_demo,
            "queue_size": _job_queue.qsize(),
        })

        try:
            await _run_pipeline(
                channel=job.channel,
                user_message=job.user_message,
                progress_msg=job.progress_msg,
                media_path=job.media_path,
                display_name=job.display_name,
                is_demo=job.is_demo,
            )
            log.info("pipeline job completed", extra={"file": job.display_name})
        except Exception as exc:
            next_attempt = job.attempt + 1
            log.error("pipeline job failed", exc_info=True, extra={
                "file": job.display_name, "attempt": next_attempt, "error": str(exc),
            })
            if next_attempt < MAX_RETRIES:
                job.attempt = next_attempt
                await _job_queue.put(job)
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
        finally:
            _job_queue.task_done()

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

    global _worker_task
    _worker_task = asyncio.create_task(_worker(), name="pipeline-worker")

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

    safe_name = attachment.filename.replace(" ", "_")
    dest = config.UPLOAD_DIR / f"{int(time.time())}_{safe_name}"

    log.info("attachment received", extra={
        "file": attachment.filename,
        "size_mb": round(attachment.size / 1e6, 2),
        "channel": str(message.channel),
        "author": str(message.author),
    })
    await attachment.save(dest)

    try:
        await message.add_reaction(REACT_ACK)
    except discord.HTTPException:
        pass

    progress_msg = await message.reply(
        f"\N{Clapper Board} Received **{attachment.filename}**. "
        "Initializing Gemma 4 and TRIBE v2 pipeline..."
    )

    job = _PipelineJob(
        media_path=dest,
        display_name=attachment.filename,
        channel=message.channel,
        user_message=message,
        progress_msg=progress_msg,
        is_demo=False,
    )
    await _job_queue.put(job)
    log.info("job enqueued", extra={"file": attachment.filename,
                                    "queue_size": _job_queue.qsize()})


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
        )
        await _job_queue.put(job)
        log.info("demo job enqueued", extra={"file": config.DEMO_VIDEO.name})
    except Exception as exc:
        log.error("demo command failed", exc_info=True)
        await interaction.followup.send(f"Could not start demo: `{exc}`")


@tree.command(name="jemma-status", description="GPU, queue, and model health.")
async def cmd_status(interaction: discord.Interaction) -> None:
    gpu = _query_nvidia_smi()
    queue_depth = _job_queue.qsize()
    worker_alive = bool(_worker_task and not _worker_task.done())

    if gpu is None:
        gpu_line = "GPU: unavailable (nvidia-smi not found)"
    else:
        vram, util, temp, power = gpu
        gpu_line = (f"GPU VRAM: **{vram:.1f} GB** | util **{util}%** | "
                    f"temp **{temp}°C** | **{power:.0f} W**")

    msg = (
        f"\N{Brain} **Jemma status**\n"
        f"{gpu_line}\n"
        f"Ollama: `{config.OLLAMA_URL}` "
        f"(fast: `{config.OLLAMA_MODEL_FAST}`, quality: `{config.OLLAMA_MODEL_QUALITY}`)\n"
        f"TRIBE weights: `{config.WEIGHTS_DIR.name}` | "
        f"`duration_trs={config.TRIBE_CONFIG_UPDATE['data.duration_trs']}`\n"
        f"Pipeline queue: **{queue_depth}** pending | "
        f"worker: **{'running' if worker_alive else 'stopped'}**\n"
        f"All inference is local — no data leaves this machine."
    )
    await interaction.response.send_message(msg, ephemeral=True)


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

    # ── Stage C: full multimodal TRIBE ────────────────────────────────────────
    t_c = time.time()
    log.debug("stage C: full multimodal TRIBE", extra={"file": display_name})
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

    # ── Tier narration ────────────────────────────────────────────────────────
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
        state.full_status = "Generating three-tier narrations (Gemma 4)..."
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

    # ── Post main embed ───────────────────────────────────────────────────────
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
            f"Dominant: **{_analysis._YEO7_FULL.get(brain_analysis.dominant_network, brain_analysis.dominant_network)}**\n\n"
            f"Top Yeo-7 networks:\n{net_lines}"
        ),
        colour=discord.Colour.from_rgb(88, 101, 242),
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
        "reply_id": result_msg.id,
    })

    # ── Demo mode: post the remaining 6 tiers as a series of follow-up messages
    if is_demo:
        await _post_demo_tiers(result_msg, all_tiers, display_name)

    log.info("pipeline complete", extra={
        "file": display_name, "total_secs": round(time.time() - t_total, 1),
    })


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
