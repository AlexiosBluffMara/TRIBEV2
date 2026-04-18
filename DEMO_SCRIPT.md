# YouTube demo script — Jemma offline brain-response Discord bot

Goal: a ~3-minute screen recording showing a cat video go into a Discord
channel, get analyzed by Gemma 4 E4B, predicted by TRIBE v2 on the local
RTX 5090, and come back with a cortical activation map and a clinician
narration. Nothing leaves the machine.

## Before you hit record

1. **Hardware warmup**
   ```bash
   nvidia-smi          # confirm the 5090 is idle (~5 GB resident for Windows compositor)
   ollama ps           # should be empty or show gemma4:e4b-it-q8_0
   ```

2. **Kick the Ollama model into VRAM once** so the first recorded request
   doesn't eat ~10s of model load time:
   ```bash
   curl -s -X POST http://localhost:11434/api/generate \
     -d '{"model":"gemma4:e4b-it-q8_0","prompt":"hello","stream":false,"think":false}' \
     > /dev/null
   ```

3. **Pre-warm TRIBE** (optional but recommended — cuts ~6s off the demo):
   ```bash
   python -c "from bot.pipeline import load_model; load_model()"
   ```
   Then exit the Python process so the `bot.py` run loads it fresh on camera
   (or keep it resident if you plan to use `run_demo.py` instead of the
   Discord flow).

4. **Regenerate `assets/cat_demo_20s.mp4`** (captures your real cat clip):
   ```bash
   python -m bot.make_demo_asset
   ```
   Look at the file in Windows Explorer to confirm it's ~3-4 MB and plays.

5. **Stage three terminals** for the recording:
   - Terminal 1: `nvidia-smi -l 1` (live GPU strain — shows up on-camera)
   - Terminal 2: `python -m bot.bot` (the Discord bot)
   - Terminal 3: `python -m bot.run_demo --skip-stream` (fallback if
     Discord goes sideways)

## Recording outline (target: 3:00)

### 0:00 — 0:20  cold open

- Quick title slide: **"Jemma — brain response inference, 100% offline"**
- B-roll: `nvidia-smi` output showing the idle RTX 5090 + `ollama list`
  showing `gemma4:e4b-it-q8_0`.

### 0:20 — 0:50  problem framing

- One sentence on why: *"Medical offices can't send patient video to the
  cloud. Our box ships with Gemma 4 and TRIBE v2 locally — no data leaves."*
- Show `docker-compose` is NOT running (emphasize single-box deploy).
- Show the Discord server with the bot online and `/jemma-demo` in the
  command list.

### 0:50 — 1:15  upload the cat video

- Drag `assets/cat_source.mp4` into the Discord channel.
- The bot replies almost immediately with **Gemma vision** text describing
  the cat clip. (~1-2 s — this is the "wow, it saw that?" moment.)
- Cut to `nvidia-smi` terminal — show VRAM briefly spike to ~18 GB during
  Gemma, then drop as `keep_alive: 0` evicts it.

### 1:15 — 2:30  TRIBE v2 inference

- Bot posts: "Running TRIBE v2 on <clip> — ~1-8 min on the 5090..."
- Fast-forward this section (2-4x speed) with a timer overlay. Cut to
  `nvidia-smi` at the halfway point — VRAM ~31 GB, GPU util ~98%,
  500+ W power draw. That's the visual proof the 5090 is working hard.
- Voice-over: *"This is V-JEPA2 + wav2vec-BERT + Llama-3.2-3B running as
  frozen feature stacks, plus an 8-layer transformer head predicting BOLD
  activity on 20,484 fsaverage5 cortical vertices at 2 Hz."*

### 2:30 — 3:00  payoff

- Bot delivers the final message: cortex heatmap PNG + Gemma narration
  paragraph. Zoom in on the image.
- Voice-over: *"The model predicted visual processing networks and dorsal
  attention — consistent with the cat moving across the frame. Gemma
  explains it in language a clinician can paste straight into a note."*
- End card: *"github.com/<you>/tribev2 - all weights local, Apache 2.0
  source, Gemma 4 Good Hackathon 2026."*

## If something goes wrong on camera

- **Bot times out on Discord**: fall back to `python -m bot.run_demo` in
  Terminal 3 — same pipeline, terminal output only. Still shows the
  hardware report at the end, which is actually a nice bonus.
- **Ollama returns empty response**: ensure `think: false` is set (Gemma 4
  is a thinking model; see `bot/gemma.py`). If it's already set, bump
  `num_predict` to 800.
- **VRAM OOM**: drop to `gemma4:e4b` (Q4_K_M, 9.6 GB) instead of
  `gemma4:e4b-it-q8_0` (11 GB). Edit `bot/config.py` `OLLAMA_MODEL`.

## Post-recording checklist

- [ ] Clip to exactly 3:00 (or shorter — attention-span budget)
- [ ] Add title card + end card
- [ ] Redact Discord server name / username if not public
- [ ] Upload at 1080p, chaptered (`problem`, `upload`, `TRIBE`, `result`)
- [ ] Link in YouTube description: repo, notebook, TRIBE v2 paper, Gemma
      model card
