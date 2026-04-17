"""Central config for the Jemma Discord bot.

Paths are resolved relative to the project root (D:\\TRIBEV2).
Secrets come from environment variables, never from disk.
"""
from __future__ import annotations

import os
import sys
import pathlib
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "tribev2_weights"
SOURCE_DIR = PROJECT_ROOT / "tribev2_src"
CACHE_DIR = PROJECT_ROOT / "tribev2_cache"
OUT_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

for d in (CACHE_DIR, OUT_DIR, ASSETS_DIR, UPLOAD_DIR):
    d.mkdir(exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    tok = PROJECT_ROOT / ".hf_token"
    if tok.exists():
        HF_TOKEN = tok.read_text().strip()
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN or ""
os.environ["HF_TOKEN"] = HF_TOKEN or ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Two-model split: a fast, small Gemma for gate/classify/progress (short calls),
# and a quality model for the final three-tier narrations. They can be the same
# model if you want; Unsloth UD-Q4_K_XL is the recommended co-residence pick.
OLLAMA_MODEL_FAST = os.environ.get("OLLAMA_MODEL_FAST", "gemma4:e4b-it-q8_0")
OLLAMA_MODEL_QUALITY = os.environ.get("OLLAMA_MODEL_QUALITY", "gemma4:e4b-it-q8_0")
# Back-compat alias used by older modules.
OLLAMA_MODEL = OLLAMA_MODEL_QUALITY

# TRIBE inference-time overrides.
# duration_trs must match the model's internal keep-mask length (100).
# Setting it lower causes an IndexError in demo_utils.predict().
TRIBE_CONFIG_UPDATE = {
    "data.duration_trs": int(os.environ.get("TRIBE_DURATION_TRS", "100")),
}

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DISCORD_GUILD_ID = os.environ.get("DISCORD_GUILD_ID")
# Only respond in this channel if set (optional guardrail).
DISCORD_ALLOWED_CHANNEL_ID = os.environ.get("DISCORD_ALLOWED_CHANNEL_ID")

DEMO_VIDEO = ASSETS_DIR / "cat_demo_20s.mp4"
