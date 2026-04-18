# Jemma — Full Setup Guide

This guide walks you through getting Jemma running from zero on Windows, Linux,
or macOS. Jemma is a local-only Discord bot: cat videos in, cortical brain-maps
out. No data leaves your machine.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Get a Discord bot token](#2-get-a-discord-bot-token)
3. [Get a Hugging Face token](#3-get-a-hugging-face-token)
4. [Install Ollama + Gemma 4 E4B](#4-install-ollama--gemma-4-e4b)
5. [Project setup](#5-project-setup)
6. [Configure .env](#6-configure-env)
7. [First run](#7-first-run)
8. [Docker setup (cross-platform alternative)](#8-docker-setup-cross-platform)
9. [Live log watcher](#9-live-log-watcher)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3090 (24 GB VRAM) | RTX 5090 (32 GB VRAM) |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB free | 256 GB NVMe SSD |

> **Mac / CPU fallback**: TRIBE v2 can run on CPU (Apple Silicon MPS works too),
> but Stage C will take 30-60 min instead of 7 min. Gemma runs via Ollama on CPU
> at reduced throughput. Not recommended for interactive use.

### Software

**Windows 11**
- NVIDIA driver 560+ and CUDA 12.8 runtime
- [Python 3.11](https://www.python.org/downloads/) — check "Add to PATH"
- [Git for Windows](https://git-scm.com/download/win) (includes git-bash)
- [ffmpeg](https://ffmpeg.org/download.html) — add `ffmpeg.exe` to PATH
  - Quick option: `winget install Gyan.FFmpeg`
- [Ollama for Windows](https://ollama.com/download)

**Linux (Ubuntu 22.04+)**
```bash
# CUDA 12.8 runtime (if not already installed via driver package)
# See: https://developer.nvidia.com/cuda-downloads

sudo apt-get install -y python3.11 python3.11-venv ffmpeg git
```

**macOS (Apple Silicon)**
```bash
brew install python@3.11 ffmpeg git
# Ollama: https://ollama.com/download (macOS pkg)
```

---

## 2. Get a Discord bot token

You need your own bot application. Takes ~5 minutes.

1. Go to **https://discord.com/developers/applications** and sign in.
2. Click **New Application** → give it a name (e.g. "Jemma") → **Create**.
3. Left sidebar → **Bot** → click **Reset Token** → copy the token.
   Save it — you won't see it again (you can always reset again).
4. On the same **Bot** page, scroll down to **Privileged Gateway Intents**:
   - Enable **Message Content Intent** ← required; the bot reads attachments.
5. Left sidebar → **OAuth2** → **URL Generator**:
   - Scopes: check `bot` and `applications.commands`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Attach Files`,
     `Add Reactions`, `Read Message History`
   - Copy the generated URL at the bottom and open it in a browser.
   - Select your server and click **Authorize**.
6. To get your **Guild (Server) ID** (optional but recommended for fast slash-
   command sync):
   - Discord → Settings → Advanced → enable **Developer Mode**.
   - Right-click your server icon → **Copy Server ID**.
7. To get a **Status Channel ID** (where Jemma posts online/offline notices):
   - Create a channel like `#jemma-status`.
   - Right-click the channel → **Copy Channel ID**.

---

## 3. Get a Hugging Face token

Jemma downloads TRIBE v2 weights and Llama-3.2-3B from Hugging Face.
Both require accepting license agreements.

1. Create a free account at **https://huggingface.co** if you don't have one.
2. Go to **Settings → Access Tokens** → **New token** →
   name it "jemma", type **Read** → **Generate token** → copy it.
3. Accept the TRIBE v2 model license:
   - Visit the TRIBE v2 repository and click **Agree and access repository**.
4. Accept the Llama 3.2 license:
   - Go to `meta-llama/Llama-3.2-3B` on HF → click through Meta's license.

The first TRIBE run will download:
- V-JEPA2 ViT-g checkpoint (~6 GB, 843 shards)
- wav2vec-BERT (~1.2 GB)
- Llama-3.2-3B (~6 GB)
- Schaefer-400 atlas and fsaverage5 mesh (auto-downloaded by nilearn)

Total: ~14 GB on first run. Subsequent runs use the local cache instantly.

---

## 4. Install Ollama + Gemma 4 E4B

**Windows**
```powershell
# Install Ollama from https://ollama.com/download, then:
ollama pull gemma4:e4b-it-q8_0    # ~7.5 GB
```

**Linux / macOS**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e4b-it-q8_0
```

**Verify it works:**
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:e4b-it-q8_0",
  "prompt": "Say hello in one sentence.",
  "think": false,
  "stream": false
}'
```
You should see a `response` field. If it's empty and `done_reason` is `"length"`,
you forgot `"think": false` — Gemma 4 is a thinking model by default.

**Optional — smaller Unsloth GGUF (saves ~2.4 GB VRAM):**
```bash
# Download UD-Q4_K_XL GGUF (~5.1 GB) and register with Ollama
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-UD-Q4_K_XL.gguf --local-dir .

cat > Modelfile <<'EOF'
FROM ./gemma-4-E4B-it-UD-Q4_K_XL.gguf
TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""
PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.4
PARAMETER num_ctx 8192
EOF

ollama create jemma-gemma -f Modelfile
# Then set OLLAMA_MODEL_FAST=jemma-gemma OLLAMA_MODEL_QUALITY=jemma-gemma in .env
```

---

## 5. Project setup

```bash
git clone <this-repo-url> TRIBEV2
cd TRIBEV2

# Create venv
python3.11 -m venv .venv

# Activate
# Windows git-bash / PowerShell:
source .venv/Scripts/activate   # git-bash
# .venv\Scripts\Activate.ps1    # PowerShell
# Linux / macOS:
source .venv/bin/activate

# Install PyTorch with CUDA 12.8 wheels first
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128

# Install remaining deps
pip install -r requirements.txt

# Install TRIBE v2 source (editable, no-deps)
pip install --no-deps -e tribev2_src/
```

> **macOS / CPU-only**: replace the torch install line with:
> ```bash
> pip install torch torchvision
> ```

---

## 6. Configure .env

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```ini
# Required
DISCORD_TOKEN=your-discord-bot-token-here
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Where your HF model cache lives — set to a fast SSD with 20+ GB free
HF_HOME=~/.cache/huggingface        # Linux/Mac default
# HF_HOME=D:\unsloth\hf_cache       # Windows example

# Fast slash-command sync (paste your server ID)
DISCORD_GUILD_ID=123456789012345678

# Optional: restrict bot to one channel
DISCORD_ALLOWED_CHANNEL_ID=

# Optional: online/offline announcements channel
DISCORD_STATUS_CHANNEL_ID=123456789012345679

# Optional: Claude API key for live log diagnosis (watch_logs.py)
ANTHROPIC_API_KEY=sk-ant-xxxx

# Ollama model (defaults work if you pulled gemma4:e4b-it-q8_0)
OLLAMA_MODEL_FAST=gemma4:e4b-it-q8_0
OLLAMA_MODEL_QUALITY=gemma4:e4b-it-q8_0
OLLAMA_URL=http://localhost:11434

# Must stay at 100 — matches TRIBE v2 training-time keep-mask length
TRIBE_DURATION_TRS=100
```

---

## 7. First run

```bash
# Optional: build the packaged 20 s demo clip (requires a source clip)
python -m bot.make_demo_asset

# Windows PowerShell:
.\start_bot.ps1

# Linux / macOS:
bash start_bot.sh
```

Watch the console for:
```
[INF] Jemma starting up
[INF] logged in  user=Jemma#1234  id=...
[INF] commands synced to guild  guild=...
[INF] pre-warming TRIBE v2 (expect ~6 s)...
[INF] TRIBE v2 pre-warmed; Jemma is ready
[INF] pipeline worker started
```

In Discord:
- `/jemma-demo` — run the packaged cat clip (no upload needed)
- Drop a `.mp4 .mov .mkv .webm .wav .mp3 .flac` file in the channel
- `/jemma-status` — GPU, queue depth, worker state

**Structured logs** are written to `logs/jemma.jsonl` (10 MB × 5 rotations).

---

## 8. Docker setup (cross-platform)

Use this if you want a reproducible environment or are running on a fresh Linux
server.

### Prerequisites for Docker

**Linux:**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com | sh

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows:** Docker Desktop 4.26+ with WSL2 backend handles GPU passthrough
automatically if NVIDIA drivers are installed.

**macOS:** GPU passthrough is not supported in Docker on macOS. Use native setup
(section 5–7) or run without GPU (will be very slow).

### Run with Docker Compose

```bash
# Build the image
docker compose build

# Pull Gemma 4 into the Ollama volume (first time only)
docker compose run --rm ollama ollama pull gemma4:e4b-it-q8_0

# Start both services
docker compose up -d

# Follow Jemma logs
docker compose logs -f jemma

# Stop everything
docker compose down
```

The `docker-compose.yml` uses these host paths (override via environment):
| Variable | Default | Purpose |
|----------|---------|---------|
| `TRIBEV2_WEIGHTS` | `./tribev2_weights` | TRIBE v2 weights (read-only mount) |
| `TRIBEV2_SRC` | `./tribev2_src` | TRIBE v2 Python source (read-only) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache |

---

## 9. Live log watcher

`watch_logs.py` tails `logs/jemma.jsonl`, batches errors, and calls Claude
claude-sonnet-4-6 to diagnose them. Requires `ANTHROPIC_API_KEY` in `.env`.

```bash
# Install anthropic if not already present
pip install anthropic

# Start watching (from end of file)
python -m bot.watch_logs

# Replay full log history then follow
python -m bot.watch_logs --from-start

# Tighter error batching (5 s instead of 10)
python -m bot.watch_logs --batch-window 5
```

When an error is detected, Claude will:
1. Identify the root cause.
2. Output a `PATCH` block if a code fix is possible.
3. In an interactive terminal, prompt `Apply this patch? [y/N]`.

The watcher never modifies files autonomously — it always asks first.

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'discord'`
Your `.venv` is not activated, or you're using the wrong Python. Check:
```bash
which python   # should point to .venv/bin/python or .venv/Scripts/python
python -c "import discord; print(discord.__version__)"
```
On Windows, hardcode the full path in `start_bot.ps1`:
```powershell
$python = "C:\path\to\TRIBEV2\.venv\Scripts\python.exe"
```

### `[WinError 87] The parameter is incorrect`
ffmpeg/ffprobe/nvidia-smi called without `CREATE_NO_WINDOW` from asyncio thread
pool. This is already fixed in `gemma_vision.py` and `hwmon.py`. If you see it
elsewhere, add `creationflags=subprocess.CREATE_NO_WINDOW` to that subprocess
call.

### `IndexError: boolean index did not match ... size 100 but ... size 50`
`TRIBE_DURATION_TRS` is set to 50 (or something other than 100). The training-
time keep-mask has exactly 100 entries and cannot be changed. Set it back to 100:
```ini
TRIBE_DURATION_TRS=100
```

### Empty Gemma responses (`done_reason: "length"`)
Ollama's Gemma 4 defaults to thinking mode and exhausts its token budget before
generating a response. The bot always passes `"think": false` — if you're calling
Ollama directly for testing, add that flag.

### Stage C very slow (>20 min)
Normal on first run (model download). After that, check:
- VRAM usage: `nvidia-smi` — if >31 GB, Gemma wasn't evicted. The bot sets
  `keep_alive: 0` so Ollama should evict after each call.
- CPU fallback: if `nvidia-smi` shows 0% GPU utilization during Stage C, PyTorch
  isn't using CUDA. Check `torch.cuda.is_available()`.

### Discord slash commands not appearing
Without a Guild ID, commands take up to 1 hour to propagate globally. Set
`DISCORD_GUILD_ID` in `.env` for instant sync to your server.

### `FileNotFoundError: ffmpeg` / `ffprobe`
Both must be on PATH. Test: `ffmpeg -version` and `ffprobe -version`.
On Windows: `winget install Gyan.FFmpeg` then restart your terminal.

### HF download fails (401 Unauthorized)
Your `HF_TOKEN` doesn't have access to the gated model. Visit the model page on
Hugging Face, accept the license, and try again. If using a Read token, make sure
it was created after you accepted the license.
