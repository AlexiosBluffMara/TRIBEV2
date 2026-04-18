"""Pre-flight dependency and environment check for Jemma.

Run this before starting the bot for the first time to verify everything
is in place. Prints a clear pass/fail for every requirement.

Usage:
    python check_setup.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
WARN = "\033[33m[WARN]\033[0m"
INFO = "\033[36m[INFO]\033[0m"

errors = 0


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    global errors
    if ok:
        print(f"{PASS} {label}{(' — ' + detail) if detail else ''}")
    elif warn_only:
        print(f"{WARN} {label}{(' — ' + detail) if detail else ''}")
    else:
        print(f"{FAIL} {label}{(' — ' + detail) if detail else ''}")
        errors += 1
    return ok


def try_import(mod: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        return True, ver
    except ImportError as e:
        return False, str(e)


# ── Python version ─────────────────────────────────────────────────────────────
check(
    "Python 3.11+",
    sys.version_info >= (3, 11),
    f"{sys.version.split()[0]}",
)

# ── Core Python packages ───────────────────────────────────────────────────────
for pkg, name in [
    ("discord",      "discord.py"),
    ("torch",        "PyTorch"),
    ("nilearn",      "nilearn"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("requests",     "requests"),
    ("psutil",       "psutil"),
    ("anthropic",    "anthropic (optional)"),
    ("yt_dlp",       "yt-dlp (optional)"),
]:
    ok, ver = try_import(pkg)
    warn = pkg in ("anthropic", "yt_dlp")
    check(f"{name}", ok, ver, warn_only=warn)

# ── CUDA ───────────────────────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        name_0 = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("CUDA available", True, f"{n_gpus} GPU(s) — {name_0} {vram_gb:.1f} GB")
    else:
        check("CUDA available", False, "torch.cuda.is_available() → False")
except Exception as e:
    check("CUDA available", False, str(e))

# ── ffmpeg + ffprobe ───────────────────────────────────────────────────────────
for tool in ("ffmpeg", "ffprobe"):
    path = shutil.which(tool)
    if path:
        try:
            ver_out = subprocess.check_output(
                [tool, "-version"], stderr=subprocess.STDOUT, text=True
            ).splitlines()[0]
            check(tool, True, ver_out[:60])
        except Exception:
            check(tool, True, path)
    else:
        check(tool, False, "not found on PATH — install ffmpeg")

# ── Ollama reachable ───────────────────────────────────────────────────────────
ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
try:
    import requests as _req
    r = _req.get(ollama_url, timeout=3)
    check("Ollama reachable", r.status_code == 200, ollama_url)
except Exception as e:
    check("Ollama reachable", False, f"{ollama_url} — {e}")

# ── .env file ──────────────────────────────────────────────────────────────────
env_path = PROJECT / ".env"
check(".env file exists", env_path.exists(),
      str(env_path) if env_path.exists() else "copy .env.example → .env")

if env_path.exists():
    env_vars = {}
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip()

    for var, required in [
        ("DISCORD_TOKEN", True),
        ("HF_TOKEN",      True),
        ("HF_HOME",       True),
        ("OLLAMA_URL",    False),
    ]:
        val = env_vars.get(var, "")
        is_set = bool(val) and "xxx" not in val.lower() and "your-" not in val.lower()
        check(f".env: {var}", is_set, "(set)" if is_set else "not set",
              warn_only=not required)

# ── TRIBE weights ──────────────────────────────────────────────────────────────
weights_dir = PROJECT / "tribev2_weights"
check(
    "TRIBE weights directory",
    weights_dir.exists() and any(weights_dir.iterdir()),
    str(weights_dir),
)

# ── TRIBE source ───────────────────────────────────────────────────────────────
src_dir = PROJECT / "tribev2_src"
check("TRIBE source directory", src_dir.exists(), str(src_dir))
ok, _ = try_import("tribev2")
check("tribev2 importable", ok, "run: pip install --no-deps -e tribev2_src/")

# ── Demo clip ──────────────────────────────────────────────────────────────────
demo = PROJECT / "assets" / "demo_clip_20s.mp4"
check(
    "Demo clip",
    demo.exists(),
    f"{demo.stat().st_size // 1024} KB" if demo.exists() else
    "run: python -m bot.make_demo_asset",
)

# ── HF cache ───────────────────────────────────────────────────────────────────
hf_home = Path(os.environ.get("HF_HOME") or
               env_vars.get("HF_HOME") if env_path.exists() else "" or
               Path.home() / ".cache" / "huggingface")
check(
    "HF cache directory",
    Path(hf_home).exists(),
    str(hf_home),
    warn_only=True,
)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
if errors == 0:
    print(f"{PASS} All required checks passed — you can start the bot.")
else:
    print(f"{FAIL} {errors} check(s) failed — fix the issues above before starting.")
sys.exit(0 if errors == 0 else 1)
