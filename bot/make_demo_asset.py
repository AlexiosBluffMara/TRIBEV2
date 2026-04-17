"""Generate the 20-second cat demo asset.

Preferred path: a real cat clip at ``assets/cat_source.mp4``. We trim the
first 20 seconds, keep its native audio (purring/meowing) so TRIBE v2's
wav2vec-bert extractor has real signal, and optionally mix in a short
narration line for TRIBE's text extractor.

Fallback path (no source clip): a synthetic gradient backdrop + gTTS
narration, so the pipeline still runs end-to-end.

Usage:
    python -m bot.make_demo_asset
    python -m bot.make_demo_asset --no-narration   # native audio only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

NARRATION = (
    "A tabby cat sits on a sunny windowsill, slowly stretching its paws. "
    "It blinks at a bird flitting past the glass, then lets out a soft purr. "
    "The rhythmic sound fills the quiet room as the cat curls back into a "
    "warm patch of afternoon light and closes its eyes."
)
CLIP_SECONDS = 20

ASSETS = config.ASSETS_DIR
OUT_VIDEO = ASSETS / "cat_demo_20s.mp4"
NARRATION_WAV = ASSETS / "cat_demo_20s_audio.wav"
SOURCE_CLIP = ASSETS / "cat_source.mp4"  # optional real cat video


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found on PATH — install ffmpeg and retry.")
    return ffmpeg


def _generate_narration() -> Path:
    from gtts import gTTS
    mp3 = ASSETS / "_narration.mp3"
    print(f"[asset] Generating narration ({len(NARRATION)} chars) via gTTS ...")
    gTTS(text=NARRATION, lang="en", slow=False).save(str(mp3))
    ffmpeg = _require_ffmpeg()
    subprocess.check_call([
        ffmpeg, "-y", "-i", str(mp3),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(NARRATION_WAV),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp3.unlink(missing_ok=True)
    return NARRATION_WAV


def _render_synthetic_video(audio: Path) -> Path:
    """Render a 20s placeholder MP4 with drawn text + the narration."""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i",
        "gradients=s=640x360:d=20:duration=20:speed=0.05:c0=0xf7a072:c1=0x5c374c,format=yuv420p",
        "-i", str(audio),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        "-t", "20",
        str(OUT_VIDEO),
    ]
    print(f"[asset] Rendering synthetic backdrop -> {OUT_VIDEO}")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return OUT_VIDEO


def _trim_real_clip(source: Path, with_narration: bool) -> Path:
    """Trim first CLIP_SECONDS of the real clip; optionally mix narration over the native audio."""
    ffmpeg = _require_ffmpeg()
    if with_narration:
        audio = _generate_narration()
        # 80% native audio (purring), 50% narration, downmix to mono 16kHz
        cmd = [
            ffmpeg, "-y",
            "-i", str(source),
            "-i", str(audio),
            "-t", str(CLIP_SECONDS),
            "-filter_complex",
            "[0:a]volume=0.8[a0];[1:a]volume=0.5,apad=pad_dur=1[a1];"
            "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[aout];"
            "[0:v]scale=480:-2,setsar=1[vout]",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "96k", "-ar", "16000", "-ac", "1",
            str(OUT_VIDEO),
        ]
        print(f"[asset] Trim {source.name} (0-{CLIP_SECONDS}s) + mix narration -> {OUT_VIDEO}")
    else:
        cmd = [
            ffmpeg, "-y",
            "-i", str(source),
            "-t", str(CLIP_SECONDS),
            "-vf", "scale=480:-2,setsar=1",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "96k", "-ar", "16000", "-ac", "1",
            str(OUT_VIDEO),
        ]
        print(f"[asset] Trim {source.name} (0-{CLIP_SECONDS}s, native audio only) -> {OUT_VIDEO}")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Also write a 16kHz mono WAV for quick inspection / TRIBE audio feature debugging.
    subprocess.check_call(
        [ffmpeg, "-y", "-i", str(OUT_VIDEO),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(NARRATION_WAV)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return OUT_VIDEO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-narration", action="store_true",
                        help="Use only the video's native audio (no gTTS overlay).")
    args = parser.parse_args()

    if SOURCE_CLIP.exists():
        _trim_real_clip(SOURCE_CLIP, with_narration=not args.no_narration)
    else:
        audio = _generate_narration()
        _render_synthetic_video(audio)
    size_mb = OUT_VIDEO.stat().st_size / 1e6
    print(f"[asset] Done - {OUT_VIDEO} ({size_mb:.1f} MB)")
    print(f"[asset] Reference WAV: {NARRATION_WAV}")


if __name__ == "__main__":
    main()
