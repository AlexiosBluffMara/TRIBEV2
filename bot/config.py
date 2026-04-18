"""Central config for the Jemma Discord bot — JemmaBrain TRIBE v2.

All paths resolved relative to project root (D:\\TRIBEV2).
Secrets come from environment variables (loaded from .env by the startup script).

Model tier architecture:
  FAST   → gemma4:e4b         (always warm, gate/classify/quick narration)
  DEEP   → gemma4:26b         (MoE, loaded on demand, tiers 0-4 analysis)
  EXPERT → gemma4:31b         (dense, Researcher role only, tiers 5-6)

Discord RBAC:
  @Guest      → rate-limited (1/4h), FAST model only, queue priority 3
  @Verified   → medium (1/h),  DEEP model,   queue priority 2
  @Researcher → high   (5/h),  EXPERT model, queue priority 1
  @Staff      → unlimited,     EXPERT model, queue priority 0
"""
from __future__ import annotations

import os
import sys
import pathlib
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = PROJECT_ROOT / "tribev2_weights"
SOURCE_DIR   = PROJECT_ROOT / "tribev2_src"
CACHE_DIR    = PROJECT_ROOT / "tribev2_cache"
OUT_DIR      = PROJECT_ROOT / "outputs"
ASSETS_DIR   = PROJECT_ROOT / "assets"
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
LOGS_DIR     = PROJECT_ROOT / "logs"

for _d in (CACHE_DIR, OUT_DIR, ASSETS_DIR, UPLOAD_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)

# ── Hugging Face ───────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    _tok = PROJECT_ROOT / ".hf_token"
    if _tok.exists():
        HF_TOKEN = _tok.read_text().strip()

os.environ["HUGGING_FACE_HUB_TOKEN"]       = HF_TOKEN or ""
os.environ["HF_TOKEN"]                      = HF_TOKEN or ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Ollama / Model config ──────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Three-tier model split:
#   FAST   = always warm in VRAM (E4B, ~10 GB) — gate, classify, quick narration
#   DEEP   = 26B MoE (~19 GB) — standard deep analysis (tiers 0-4)
#   EXPERT = 31B dense (~21 GB) — maximum quality (tiers 5-6, researchers)
OLLAMA_MODEL_FAST   = os.environ.get("MODEL_FAST",   "gemma4:e4b")
OLLAMA_MODEL_DEEP   = os.environ.get("MODEL_DEEP",   "gemma4:26b")
OLLAMA_MODEL_EXPERT = os.environ.get("MODEL_EXPERT", "gemma4:31b")

# Back-compat aliases
OLLAMA_MODEL_QUALITY = OLLAMA_MODEL_DEEP    # used by tiers.py
OLLAMA_MODEL         = OLLAMA_MODEL_QUALITY

# Flash attention in Ollama (reduces VRAM at long contexts on Blackwell)
OLLAMA_FLASH_ATTENTION = os.environ.get("OLLAMA_FLASH_ATTENTION", "1")
os.environ["OLLAMA_FLASH_ATTENTION"] = OLLAMA_FLASH_ATTENTION

# KV cache quantization (q8_0 halves KV cache VRAM at minimal quality cost)
OLLAMA_KV_CACHE_TYPE = os.environ.get("OLLAMA_KV_CACHE_TYPE", "q8_0")
os.environ["OLLAMA_KV_CACHE_TYPE"] = OLLAMA_KV_CACHE_TYPE

# ── TRIBE v2 ───────────────────────────────────────────────────────────────────

# duration_trs must match model's keep-mask length (100 TRs = 50 seconds at 2 Hz).
# CRITICAL: do not lower this — breaks index into segment keep-mask.
TRIBE_CONFIG_UPDATE = {
    "data.duration_trs": int(os.environ.get("TRIBE_DURATION_TRS", "100")),
}

# ── Discord ────────────────────────────────────────────────────────────────────

DISCORD_TOKEN              = os.environ.get("DISCORD_TOKEN")
DISCORD_GUILD_ID           = os.environ.get("DISCORD_GUILD_ID")
DISCORD_ALLOWED_CHANNEL_ID = os.environ.get("DISCORD_ALLOWED_CHANNEL_ID")
DISCORD_STATUS_CHANNEL_ID  = os.environ.get("DISCORD_STATUS_CHANNEL_ID")
DISCORD_RESULTS_CHANNEL_ID = os.environ.get("DISCORD_RESULTS_CHANNEL_ID")  # central feed

# ── Discord RBAC role names (case-insensitive match) ──────────────────────────
# Override with comma-separated role names if your server uses different names.

RBAC_STAFF_ROLES      = set(os.environ.get("RBAC_STAFF_ROLES", "Staff,Admin,Moderator").split(","))
RBAC_RESEARCHER_ROLES = set(os.environ.get("RBAC_RESEARCHER_ROLES", "Researcher,Scientist,Neuroscientist").split(","))
RBAC_VERIFIED_ROLES   = set(os.environ.get("RBAC_VERIFIED_ROLES", "Verified,Member,Trusted,Phone Verified").split(","))

# ── Rate limits (jobs per user per rolling window) ────────────────────────────

RATE_LIMIT_GUEST_PER_HOUR      = int(os.environ.get("RATE_GUEST_PER_HOUR",      "1"))
RATE_LIMIT_VERIFIED_PER_HOUR   = int(os.environ.get("RATE_VERIFIED_PER_HOUR",   "4"))
RATE_LIMIT_RESEARCHER_PER_HOUR = int(os.environ.get("RATE_RESEARCHER_PER_HOUR", "20"))
RATE_LIMIT_STAFF_PER_HOUR      = int(os.environ.get("RATE_STAFF_PER_HOUR",      "999"))

# ── Queue priorities ──────────────────────────────────────────────────────────
# Lower number = higher priority (processed first from priority queue).
QUEUE_PRIORITY_STAFF      = 0
QUEUE_PRIORITY_RESEARCHER = 1
QUEUE_PRIORITY_VERIFIED   = 2
QUEUE_PRIORITY_GUEST      = 3

# Maximum items allowed in queue before rejecting new jobs
QUEUE_MAX_LENGTH = int(os.environ.get("QUEUE_MAX_LENGTH", "20"))

# ── Pipeline limits ───────────────────────────────────────────────────────────

MAX_UPLOAD_MB  = int(os.environ.get("MAX_UPLOAD_MB", "50"))      # increased from 25
TRIBE_MAX_SECS = float(os.environ.get("TRIBE_MAX_SECS", "50"))   # TRIBE v2 hard limit

# ── Misc ──────────────────────────────────────────────────────────────────────

DEMO_VIDEO = ASSETS_DIR / "demo_clip_20s.mp4"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Enable chain-of-thought streaming in Discord threads (requires MODEL_DEEP or EXPERT)
ENABLE_THREAD_COT = os.environ.get("ENABLE_THREAD_COT", "1") in ("1", "true", "yes")

# Post all completed results to the central results feed channel
ENABLE_RESULTS_FEED = os.environ.get("ENABLE_RESULTS_FEED", "1") in ("1", "true", "yes")
