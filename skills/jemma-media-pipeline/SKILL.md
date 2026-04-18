---
name: jemma-media-pipeline
description: Run the complete Jemma offline brain-response pipeline on any media file. Uses Gemma 4 E4B for visual scene description and seven-tier audience narration (toddler to neuroscience researcher), and TRIBE v2 for cortical BOLD prediction. Generates a peak-cortex PNG, multi-atlas analysis, and narrations for all seven expertise levels. Use when the user wants to analyse a video/audio clip for brain response and get explanations for multiple audiences.
license: MIT (pipeline code). CC-BY-NC 4.0 (TRIBE v2 weights). Gemma 4 model terms apply.
compatibility: Requires Python 3.11+, PyTorch CUDA, Ollama running gemma4:e4b-it-q8_0, ffmpeg, the TRIBE v2 checkpoint. Run from the TRIBEV2 project root.
metadata:
  author: jemma-tribev2
  version: "2.0"
  gemma-model: gemma4:e4b-it-q8_0
  tribe-model: TRIBE v2 (CC-BY-NC 4.0)
  hackathon: Gemma 4 Good Hackathon 2026 (Kaggle/Google DeepMind)
---

# Jemma Full Media Pipeline

Jemma is a fully offline, multimodal brain-response analysis system built on
**Gemma 4 E4B** (via Ollama) and **TRIBE v2** (local GPU inference).

## Pipeline stages

```
Media file
    │
    ▼
Stage A (~2 s)    Gemma 4 vision — keyframe description + modality classification
    │
    ▼
[Auto-trim to 50 s if needed]
    │
    ▼
Stage B (~15 s)   Text-only TRIBE fast path — language-cortex preview narration
    │
    ▼
Stage C (~4-7 min) Full multimodal TRIBE (V-JEPA2 + wav2vec-BERT + Llama-3.2-3B)
    │
    ▼
BrainAnalysis     Multi-atlas ROI projection + Yeo-7 network laterality + temporal dynamics
    │
    ▼
Tier narrations   Gemma 4 generates 7 expertise-level explanations (parallel possible)
    │
    ▼
Output            Peak PNG + structured report + 7 narrations
```

## Step-by-step

### 1. Check environment
```bash
python check_setup.py
```
All 24 checks must pass before running.

### 2. Run the full pipeline (headless)
```bash
python skills/jemma-media-pipeline/scripts/run_pipeline.py <media_path> [--all-tiers]
```

Options:
- `--all-tiers`: Generate all 7 tiers (default: tiers 2, 5, 6)
- `--demo`: Use the packaged demo clip (`assets/demo_clip_20s.mp4`)
- `--brainnetome`: Include Brainnetome-246 atlas (downloads ~50 MB on first run)

### 3. Run via the Discord bot
Start the bot and either:
- Drop a media file in the configured channel
- Use `/jemma-demo` for the packaged demo with all 7 tiers

## Audience tiers

| Tier | Audience | Prompt tokens |
|------|----------|---------------|
| 0 | Toddler / age 3–5 | 120 |
| 1 | General adult / no science background | 180 |
| 2 | Curious adult / general public | 260 |
| 3 | High school student | 300 |
| 4 | College-educated adult | 340 |
| 5 | Clinician / medical professional | 420 |
| 6 | Neuroscience researcher / ML scientist | 520 |

## Output files

| File | Content |
|------|---------|
| `outputs/brain_peak.png` | Cortex surface map at peak TR |
| `outputs/preds.npy` | Raw BOLD predictions (T, 20484) |
| `outputs/roi_schaefer400.parq` | Schaefer-400 ROI time series |
| `outputs/report.json` | Full analysis + all tier narrations |
| `outputs/gemma_vision.txt` | Gemma 4 scene description |
| `outputs/gemma_narration.txt` | All tier narrations |

## Gemma 4 roles

Gemma 4 E4B (via Ollama) handles two distinct roles:

**Vision (Stage A)**: Given keyframes extracted by ffmpeg, Gemma 4 returns a
structured JSON describing `content_type`, `subject`, `setting`, `action`,
`mood`, `modality`, and a free-text `description`. This is injected as the
stimulus label into TRIBE.

**Narration (post Stage C)**: Gemma 4 receives the full `BrainAnalysis.gemma_context()`
block (Yeo-7 network ranking, top Schaefer-400 / Harvard-Oxford / Jülich ROIs,
temporal dynamics) and generates audience-appropriate explanations. The same
brain data is narrated at 7 expertise levels without modification.

## Hackathon compliance (Gemma 4 Good Hackathon)

This project uses Gemma 4 as the **primary AI model** for both:
1. Multimodal visual understanding (scene description from keyframes)
2. Audience-adapted narration of neuroscience findings (7 tiers)

Gemma 4 is the only LLM in the pipeline. TRIBE v2 is a domain-specific
neural encoding model (not an LLM). The project runs fully offline on
consumer hardware (RTX 5090 or similar), satisfying the "limited connectivity"
and "consumer-grade hardware" requirements.

Target hackathon categories:
- **Health & Sciences** — brain-response prediction from real-world media
- **Future of Education** — 7-tier adaptive narration for any audience
- **Digital Equity & Inclusivity** — fully offline, no API costs after setup

See [references/HACKATHON.md](references/HACKATHON.md) for submission details.
