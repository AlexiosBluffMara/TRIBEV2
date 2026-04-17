"""MindCat — the cat-loving MindScope Discord bot.

Flow on every video attachment (or /mindcat-demo):

1. React :cat: on the user's message (acknowledged).
2. Post progress comment #1. Progressively edit it as each stage finishes.
3. Stage A (~2 s): Gemma vision gate — cat-classify + plain-English summary.
4. Stage B (~10-20 s): TRIBE text-only on Gemma's description -> quick
   language-cortex narration. Edit comment #1 to include it.
5. Stage C (~4-7 min): TRIBE full multimodal (video+audio+text). While
   this runs, react :brain: on comment #1.
6. When stage C finishes, post comment #2 — a Discord embed with THREE
   audience-tiered narrations (layperson / clinician / researcher) plus
   the peak-cortex PNG attached. React :white_check_mark: on comment #1.

Progress reactions on comment #1:
  :eyes:  vision done        :zap:   quick text-TRIBE done
  :brain: full TRIBE running :white_check_mark:  done
"""
from __future__ import annotations

import asyncio
import io
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands

from . import cat_gate, config, tiers
from .hwmon import _query_nvidia_smi
from .pipeline import load_model, run_inference, run_inference_text_only
from .visualize import render_peak_cortex


MAX_UPLOAD_MB = 25
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac"}

REACT_ACK = "\N{Cat Face}"             # 🐱
REACT_VISION = "\N{Eyes}"              # 👀
REACT_QUICK = "\N{High Voltage Sign}"  # ⚡
REACT_BRAIN = "\N{Brain}"              # 🧠
REACT_DONE = "\N{White Heavy Check Mark}"  # ✅
REACT_ERROR = "\N{Cross Mark}"         # ❌

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


@client.event
async def on_ready() -> None:
    print(f"[bot] logged in as {client.user} (id={client.user.id})")
    if config.DISCORD_GUILD_ID:
        guild = discord.Object(id=int(config.DISCORD_GUILD_ID))
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
        print(f"[bot] synced commands to guild {config.DISCORD_GUILD_ID}")
    else:
        await tree.sync()
        print("[bot] synced global commands (may take up to 1 h to appear)")
    # Pre-warm TRIBE so the first request is not a cold start.
    await asyncio.to_thread(load_model)
    print("[bot] TRIBE v2 pre-warmed; ready for cat videos.")


@tree.command(name="mindcat-demo", description="Run MindCat on the packaged cat clip.")
async def cmd_demo(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    try:
        start_msg = await interaction.followup.send(
            "\N{Cat Face} Running the packaged cat demo...", wait=True,
        )
        await _run_pipeline(
            channel=interaction.channel,
            user_message=None,
            progress_msg=start_msg,
            media_path=config.DEMO_VIDEO,
            display_name=config.DEMO_VIDEO.name,
        )
    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Pipeline failed: `{e}`")


@tree.command(name="mindcat-status", description="GPU and model health.")
async def cmd_status(interaction: discord.Interaction) -> None:
    gpu = _query_nvidia_smi()
    if gpu is None:
        msg = "GPU: unavailable"
    else:
        vram, util, temp, power = gpu
        msg = (
            f"\N{Cat Face} MindCat is purring.\n"
            f"GPU VRAM used: {vram:.1f} GB | util {util}% | temp {temp}C | {power:.0f} W\n"
            f"Ollama: `{config.OLLAMA_URL}` (fast: `{config.OLLAMA_MODEL_FAST}`, "
            f"quality: `{config.OLLAMA_MODEL_QUALITY}`)\n"
            f"TRIBE weights: `{config.WEIGHTS_DIR.name}` | `duration_trs="
            f"{config.TRIBE_CONFIG_UPDATE['data.duration_trs']}`\n"
            f"All inference is local — no data leaves this machine."
        )
    await interaction.response.send_message(msg, ephemeral=True)


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
            f"That clip is {attachment.size/1e6:.1f} MB — too chonky "
            f"(>{MAX_UPLOAD_MB} MB). Trim it and try again."
        )
        return

    safe_name = attachment.filename.replace(" ", "_")
    dest = config.UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    await attachment.save(dest)
    try:
        await message.add_reaction(REACT_ACK)
    except discord.HTTPException:
        pass

    progress_msg = await message.reply(
        f"\N{Cat Face} Got it — **{attachment.filename}**. Warming up "
        f"whiskers and Gemma 4...",
    )
    try:
        await _run_pipeline(
            channel=message.channel,
            user_message=message,
            progress_msg=progress_msg,
            media_path=dest,
            display_name=attachment.filename,
        )
    except Exception as e:
        traceback.print_exc()
        await progress_msg.reply(f"Pipeline failed: `{e}`")
        try:
            await progress_msg.add_reaction(REACT_ERROR)
        except discord.HTTPException:
            pass


def _is_media(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT


@dataclass
class _ProgressState:
    header: str          # top line (cat emoji + filename)
    vision: str = ""     # Gemma's visual description
    cat_remark: str = "" # one-liner from cat-gate
    quick: str = ""      # quick TRIBE narration
    full_status: str = ""  # "Running full multimodal analysis..." etc.

    def render(self) -> str:
        parts = [self.header]
        if self.vision:
            parts.append(f"\n\N{Eyes} **What I see:** {self.vision}")
        if self.cat_remark:
            parts.append(f"\n\N{Paw Prints} _{self.cat_remark}_")
        if self.quick:
            parts.append(
                f"\n\N{High Voltage Sign} **Quick read (text-only TRIBE, ~15 s):**\n{self.quick}"
            )
        if self.full_status:
            parts.append(f"\n\N{Brain} {self.full_status}")
        return "\n".join(parts)[:1990]  # Discord msg cap 2000 chars


async def _edit(progress_msg: discord.Message, state: _ProgressState) -> None:
    try:
        await progress_msg.edit(content=state.render())
    except discord.HTTPException as e:
        print(f"[bot] edit failed: {e}")


async def _react(msg: discord.Message, emoji: str) -> None:
    try:
        await msg.add_reaction(emoji)
    except discord.HTTPException:
        pass


async def _run_pipeline(
    channel: discord.abc.Messageable,
    user_message: discord.Message | None,
    progress_msg: discord.Message,
    media_path: Path,
    display_name: str,
) -> None:
    state = _ProgressState(header=f"\N{Cat Face} **{display_name}** — analyzing...")

    # --- Stage A: cat gate + vision description -------------------------
    cls = await asyncio.to_thread(cat_gate.classify, media_path)
    if cls.is_cat:
        state.header = f"\N{Cat Face} **{display_name}** — genuine cat detected \N{Paw Prints}"
    else:
        state.header = (
            f"\N{Cat Face} **{display_name}** — not a cat, but I'll analyze "
            f"anyway (I contain multitudes)"
        )
    state.vision = cls.short_description()
    state.cat_remark = cls.cat_remark
    state.full_status = ""
    await _edit(progress_msg, state)
    await _react(progress_msg, REACT_VISION)

    # --- Stage B: text-only fast TRIBE ----------------------------------
    quick_desc = cls.short_description()
    t_quick = time.time()
    quick_result = await asyncio.to_thread(run_inference_text_only, quick_desc)
    quick_text = await asyncio.to_thread(tiers.narrate_quick, quick_result, quick_desc)
    quick_secs = time.time() - t_quick
    state.quick = f"{quick_text}\n_(text-only pass, {quick_secs:.0f} s)_"
    state.full_status = (
        "Running **full multimodal** TRIBE v2 (V-JEPA2 + wav2vec-BERT + "
        "Llama-3.2-3B). This takes ~4-7 min on the 5090. Standing by..."
    )
    await _edit(progress_msg, state)
    await _react(progress_msg, REACT_QUICK)
    await _react(progress_msg, REACT_BRAIN)

    # --- Stage C: full multimodal TRIBE ---------------------------------
    t_full = time.time()
    result = await asyncio.to_thread(run_inference, media_path)
    full_secs = time.time() - t_full

    # Render peak cortex PNG
    peak_png = await asyncio.to_thread(render_peak_cortex, result)

    # Three-tier narration
    label = f"{display_name} — {cls.short_description()}"
    narr = await asyncio.to_thread(tiers.narrate_tiered, result, label)

    # --- Finalize progress comment --------------------------------------
    state.full_status = (
        f"Full analysis complete in **{full_secs/60:.1f} min**. "
        f"Peak activity at **t={result.peak_t/2:.1f}s**. See the reply below."
    )
    await _edit(progress_msg, state)
    await _react(progress_msg, REACT_DONE)

    # --- Post comment #2: the three-tier embed + cortex PNG -------------
    embed = discord.Embed(
        title=f"\N{Cat Face} Full brain-response analysis — {display_name}",
        description=(
            f"TRIBE v2 predicted **{result.preds.shape[0]}** timesteps of BOLD "
            f"activity across **{result.preds.shape[1]:,}** fsaverage5 vertices. "
            f"Peak activity: **t={result.peak_t/2:.1f}s** of "
            f"{result.preds.shape[0]/2:.1f}s total. Inference took "
            f"{full_secs/60:.1f} min on the local 5090."
        ),
        colour=discord.Colour.from_rgb(247, 160, 114),
    )
    embed.add_field(
        name="\N{Paw Prints} For a curious human (layperson)",
        value=_truncate(narr.layperson, 1020),
        inline=False,
    )
    embed.add_field(
        name="\N{Stethoscope} For a clinician",
        value=_truncate(narr.clinician, 1020),
        inline=False,
    )
    embed.add_field(
        name="\N{Microscope} For a researcher",
        value=_truncate(narr.researcher, 1020),
        inline=False,
    )
    embed.set_footer(text="TRIBE v2 (CC-BY-NC 4.0) x Gemma 4 E4B x Ollama. Offline, local, no cloud.")

    with peak_png.open("rb") as f:
        file = discord.File(io.BytesIO(f.read()), filename=peak_png.name)
    embed.set_image(url=f"attachment://{peak_png.name}")
    await progress_msg.reply(embed=embed, file=file)


def _truncate(text: str, limit: int) -> str:
    text = text.strip() or "(Gemma returned an empty response.)"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def main() -> None:
    if not config.DISCORD_TOKEN:
        raise SystemExit(
            "DISCORD_TOKEN env var is required. See bot/README.md for setup."
        )
    client.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
