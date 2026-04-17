# MindCat — offline cat-brain Discord bot

MindCat is a Discord bot that eats cat videos and returns a cortical
activity map plus three audience-tiered narrations. Everything runs on
one local box (RTX 5090, 32 GB VRAM, Windows 11). No data leaves the
machine.

The bot's "brain" is Gemma 4 E4B via Ollama. Gemma categorizes the clip,
kicks a fast text-only TRIBE pass for an immediate reaction, waits on
full multimodal TRIBE, and then translates the predicted BOLD response
into three voices: curious-human, clinician, and researcher.

## Pipeline

```
user drops a video
     |
     v  (Stage A, ~2 s)  Gemma vision on 4 keyframes -- JSON classify +
     |                    "what I see" plain-English summary
     |                   React 👀 on progress message
     v
     v  (Stage B, ~15 s) TRIBE text-only on Gemma's description -->
     |                    language-cortex quick read
     |                   React ⚡, then 🧠 (full run starts)
     v
     v  (Stage C, ~7 min) TRIBE full multimodal
     |                    V-JEPA2 (video) + wav2vec-BERT (audio) +
     |                    Llama-3.2-3B (text) frozen feature stacks
     |                      --> 8-layer transformer head
     |                      --> fsaverage5 BOLD (20,484 vertices @ 2 Hz)
     v
     v  Schaefer-400 ROI aggregation --> peak cortex PNG
     v  Three Gemma calls: layperson / clinician / researcher narration
     |
     v
Discord: progress comment gets ✅. Reply comment = embed with all three
tiers + peak-cortex image attached.
```

Progress reactions on the first bot comment:

| Emoji | Meaning |
|---|---|
| 👀 | Gemma vision done |
| ⚡ | text-only TRIBE quick read done |
| 🧠 | full multimodal TRIBE running |
| ✅ | full pipeline done, reply embed posted |
| ❌ | something went wrong (error in reply) |

And 🐱 on the user's original message as an immediate ack.

## Files

| File | Purpose |
|---|---|
| `bot/config.py` | paths, env vars, two-model split, TRIBE overrides |
| `bot/prompts.py` | central persona + all system/user prompt templates |
| `bot/ollama_client.py` | `generate()` / `generate_json()` with `think:false` + `keep_alive:0` |
| `bot/cat_gate.py` | Gemma JSON-mode classify of 4 keyframes → `CatClassification` |
| `bot/tiers.py` | three-tier (layperson/clinician/researcher) + quick narration |
| `bot/pipeline.py` | `TribeModel` loader, `run_inference`, `run_inference_text_only` |
| `bot/visualize.py` | peak cortex PNG + streaming ROI MP4 |
| `bot/gemma_vision.py` | ffmpeg keyframe extraction helpers |
| `bot/gemma.py` | legacy clinician narration for `run_demo` back-compat |
| `bot/hwmon.py` | nvidia-smi + psutil sampling around each stage |
| `bot/make_demo_asset.py` | build `assets/cat_demo_20s.mp4` from a source clip |
| `bot/run_demo.py` | CLI end-to-end run with hardware report |
| `bot/bot.py` | discord.py bot with progressive edits + reactions |

## Quick start (TL;DR)

```bash
# 0. venv
source .venv/Scripts/activate   # Windows git-bash

# 1. Ollama + Gemma 4 E4B
ollama pull gemma4:e4b-it-q8_0
# ...or Unsloth (see "Unsloth GGUFs" below)

# 2. Optional: build the packaged 20 s demo clip
python -m bot.make_demo_asset

# 3. End-to-end CLI smoke test (no Discord)
python -m bot.run_demo

# 4. Discord bot
export DISCORD_TOKEN=...
export DISCORD_GUILD_ID=...       # optional: instant slash-command sync
python -m bot.bot
```

In Discord:
- `/mindcat-demo` — runs the packaged cat clip
- Drop an `.mp4/.mov/.mkv/.webm/.wav/.mp3/.flac` in the channel — the
  bot picks it up automatically (<= 25 MB)
- `/mindcat-status` — GPU + Ollama + TRIBE config health

## Full setup walkthrough

### 1. Prereqs

- Windows 11 with WSL/git-bash or native PowerShell
- NVIDIA driver + CUDA 12 for the 5090
- Python 3.11, `uv` or `venv`
- `ffmpeg` and `ffprobe` on PATH
- Ollama for Windows: https://ollama.com/download
- A Hugging Face token with access to TRIBE v2 (CC-BY-NC 4.0) and
  Llama-3.2-3B (Meta gated)
- A Discord bot token (see step 4)

### 2. Project setup

```bash
git clone <this repo> D:\TRIBEV2
cd D:/TRIBEV2
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
# or: pip install discord.py==2.7.1 torch torchvision torchaudio \
#                 nilearn matplotlib ffmpeg-python requests

echo $HF_TOKEN > .hf_token    # or set HF_TOKEN env var
```

The first TRIBE run downloads V-JEPA2 ViT-g (~6 GB), wav2vec-BERT,
Llama-3.2-3B, and the Schaefer-400 / fsaverage5 assets into
`~/.cache/huggingface` and `~/.nilearn_data`.

### 3. Install Gemma 4 E4B in Ollama

Default (stock tag):

```bash
ollama pull gemma4:e4b-it-q8_0    # ~7.5 GB
```

See "Unsloth GGUFs" below for the smaller/better-quantized option.

Sanity check:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:e4b-it-q8_0",
  "prompt": "describe a cat",
  "think": false,
  "stream": false
}'
```

If `response` is empty and `done_reason` is `"length"`, you forgot
`"think": false`. Gemma 4 is a thinking model by default and will burn
its whole budget on hidden reasoning tokens. The bot's `ollama_client`
already sets this for you.

### 4. Discord bot token

1. https://discord.com/developers/applications → New Application →
   give it a name (e.g. "MindCat").
2. Bot tab → Reset Token → copy → `export DISCORD_TOKEN=...`.
3. Bot tab → enable **Message Content Intent** (required — the bot
   reads attachments from on_message).
4. OAuth2 → URL Generator → scopes: `bot`, `applications.commands`;
   perms: `Send Messages`, `Embed Links`, `Attach Files`,
   `Add Reactions`, `Read Message History`.
5. Visit the generated URL and invite the bot to your server.
6. Copy the guild ID (right-click server → Copy Server ID, Developer
   Mode on) → `export DISCORD_GUILD_ID=...`. Optional, but skipping it
   means slash commands take up to an hour to propagate.
7. Optional: restrict to a single channel → `export
   DISCORD_ALLOWED_CHANNEL_ID=...`.

### 5. Run

```bash
python -m bot.bot
```

First boot pre-warms TRIBE (~6 s). Watch the console for `logged in
as ...` and `TRIBE v2 pre-warmed`.

## Unsloth GGUFs via Ollama

Yes — Ollama can load any GGUF via a Modelfile. Unsloth publishes
dynamically-quantized Gemma 4 E4B GGUFs at
`unsloth/gemma-4-E4B-it-GGUF`. Recommended pick for this box is
**UD-Q4_K_XL** (~5.1 GB) — it leaves headroom for TRIBE's 30.8 GB peak.

```bash
# 1. download the GGUF (example: UD-Q4_K_XL)
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-UD-Q4_K_XL.gguf --local-dir .

# 2. write a Modelfile
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

# 3. register with Ollama
ollama create mindcat-gemma -f Modelfile

# 4. point the bot at it
export OLLAMA_MODEL_FAST=mindcat-gemma
export OLLAMA_MODEL_QUALITY=mindcat-gemma
```

You can split fast/quality across two tags — e.g. UD-Q4_K_XL for the
gate/classify and Q8_0 for the three-tier narrations — by pointing
`OLLAMA_MODEL_FAST` and `OLLAMA_MODEL_QUALITY` at different names.

## Accuracy hit

Honest accounting of every knob this project turns away from
default/full-precision:

| Knob | Setting | Accuracy impact |
|---|---|---|
| Gemma quant: Q8_0 → UD-Q4_K_XL | default in Unsloth instructions above | ~1-2% PPL on Gemma-3 benchmarks; imperceptible for narration-style outputs. Dynamic imatrix quant preserves attention and embedding layers at higher precision. |
| Gemma quant: Q8_0 → Q4_K_M (stock) | alternative | ~3-5% PPL. Still fine for three-sentence summaries; watch for minor factual drift on long clinical passages. |
| `TRIBE_DURATION_TRS` 100 → 50 | default here | Zero impact for clips ≤ 50 s. This is an inference dataloader bound, not a model capacity knob — a 20 s clip is 40 TRs, so 50 is already headroom. |
| `keep_alive: 0` on Gemma | always on | Zero accuracy impact. Pure memory hygiene — Ollama evicts the GGUF so TRIBE has room to run its ~30.8 GB peak. |
| `think: false` on Gemma | always on | Zero impact on the text content of the response. We lose the hidden reasoning trace, but the three-tier prompts don't need it and leaving it on caused empty responses (thinking budget exhausted). |
| Text-only TRIBE fast path | Stage B | **Different signal**, not a degraded version. Feeding text-only means TRIBE predicts language-cortex response to the description, not visual-cortex response to the video. The prompt explicitly flags this. The full multimodal pass in Stage C is the real result. |

Net: the quality ceiling on the narrations is set by Gemma's weights
and TRIBE v2's training data, not by anything this project quantizes.

## Hardware profile (RTX 5090, 32 GB VRAM)

Measured on a 20 s 480×854 real cat clip with mixed native + narration
audio:

| Stage | Wall | VRAM peak | GPU util | RAM peak |
|---|---|---|---|---|
| Gemma vision (4 frames) | 1.4 s | 17.9 GB | 80% | 25 GB |
| TRIBE load | 6.3 s | 19.7 GB | 80% | 27 GB |
| TRIBE text-only fast path | ~15 s | 28 GB | 95% | 34 GB |
| **TRIBE full multimodal** | **~450 s** | **30.8 GB** | **98%** | **38 GB** |
| Visualize peak PNG | 2.4 s | 18.7 GB | 8% | 28 GB |
| Gemma three-tier narration | ~6 s | 17.9 GB | 25% | 28 GB |

TRIBE full inference dominates (~7.5 min / 20 s clip). V-JEPA2's ViT-g
loads 843 weight shards and runs 40 segments at ~1.65 s each. Keeping
Gemma evicted (`keep_alive: 0`) is what pins peak at 30.8 GB instead of
overflowing.

## Storage

Per request artifacts:

| Artifact | Size |
|---|---|
| Uploaded clip | 1 - 25 MB |
| `outputs/brain_peak.png` | ~200 KB |
| `outputs/tribev2_stream.mp4` | ~110 KB |
| `outputs/roi_timeseries.parquet` | ~340 KB |
| `outputs/preds.npy` (20484 × T float32) | ~1.7 MB per 20 s |
| `tribev2_cache/` (per-video extracted features) | ~5 - 20 MB |

For a busy clinic (100 clips/day, 7-day retention): ~1-3 GB/week for
clips + derived artifacts; a 256 GB NVMe holds months uncompressed.
`tribev2_cache/` grows unbounded — prune weekly (see
`bot/cleanup.py`, coming next).

Recommended upload/storage compression (already applied in
`make_demo_asset.py`):
- H.264 CRF 23, 480p, 24 fps, 96 kbps AAC mono 16 kHz
- Archival-only: CRF 28 + 360p → ~40% smaller

## Optimization ladder

Order of operations if the ~7.5 min TRIBE full pass needs to come down:

1. **Per-clip feature cache** (already on via `tribev2_cache`) —
   re-running the same video is instant.
2. **Shorter `data.duration_trs`** — already at 50; dropping further
   only helps if your clips are shorter than ~25 s.
3. **bf16 cast of the transformer head** — the 5090 has bf16 support;
   V-JEPA2 already runs in bf16.
4. **Keep TRIBE resident in VRAM** between requests — we already do;
   load happens once per process, via `on_ready` pre-warm.
5. **Skip streaming MP4** for very short clips (`--skip-stream` in the
   CLI runner).

Do NOT try to reduce V-JEPA2 segment count — the fpc64 encoder is
baked into the ViT-g checkpoint. The real knob is `data.duration_trs`,
not a `video.num_segments` key (which doesn't exist in TRIBE's config).
