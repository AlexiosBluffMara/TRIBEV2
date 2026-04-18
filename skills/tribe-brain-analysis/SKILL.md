---
name: tribe-brain-analysis
description: Run TRIBE v2 brain-response prediction on a local media file (video, audio, or text). Returns predicted BOLD activity across 20,484 fsaverage5 cortical vertices at 2 Hz, plus multi-atlas ROI analysis (Schaefer-400, Harvard-Oxford, Jülich), Yeo-7 network laterality, temporal dynamics, and a peak-cortex PNG. Use when the user wants to predict brain responses to a video clip, audio recording, or text stimulus.
license: CC-BY-NC 4.0 (TRIBE v2 model weights). Wrapper code MIT.
compatibility: Requires Python 3.11+, PyTorch with CUDA, the TRIBE v2 checkpoint at tribev2_weights/, nilearn, numpy, pandas. Run from the TRIBEV2 project root.
metadata:
  author: jemma-tribev2
  version: "2.0"
  model: TRIBE v2 (fsaverage5, 2 Hz, 20484 vertices, 100 TR max)
  gemma-model: gemma4:e4b-it-q8_0
  allowed-tools: Bash(python:*) Read
---

# TRIBE v2 Brain-Response Analysis

TRIBE v2 (Transformer for Real-world Imagery Brain Encoding, version 2) predicts group-averaged
cortical BOLD responses to any short media stimulus using three encoders:

- **V-JEPA2** — visual feature extraction from video frames
- **wav2vec-BERT** — audio feature extraction from the sound track
- **Llama-3.2-3B** — semantic/text feature extraction from transcripts or descriptions

All inference is local. No data leaves the machine.

## Key constraints

| Parameter | Value | Reason |
|-----------|-------|--------|
| Max clip duration | **50 seconds** | `duration_trs=100` TRs at 2 Hz |
| Output shape | `(T, 20484)` | T TRs × fsaverage5 vertices |
| Hemodynamic lag | 5 s (pre-applied) | Baked into TRIBE v2 training |
| GPU required | CUDA (RTX 5090 recommended) | V-JEPA2 needs ≥ 16 GB VRAM |

If the clip is longer than 50 s, trim it first:
```bash
ffmpeg -y -i input.mp4 -t 50 -c:v libx264 -crf 23 -c:a aac trimmed.mp4
```

## Step-by-step instructions

### 1. Activate the environment
```bash
# Windows
C:\Users\soumi\TRIBEV2\.venv\Scripts\python.exe scripts/run_analysis.py <media_path>

# Linux / macOS
.venv/bin/python scripts/run_analysis.py <media_path>
```

### 2. Interpret the output
The script prints and returns a JSON summary. Key fields:

- `duration_s` — analysed clip length
- `peak_s` — time of peak mean |z| (seconds into clip)
- `peak_z` — peak mean |z| across all vertices
- `dominant_network` — top Yeo-7 network by mean activation
- `networks_ranked` — all 7 networks sorted by mean |z|
- `top_rois_schaefer400` — top 8 Schaefer-400 ROIs with mean |z|
- `top_rois_harvard_oxford` — top 8 anatomical ROIs (amygdala, thalamus, etc.)
- `activation_fraction_1sd` — fraction of cortex above 1σ at peak frame
- `cortex_png` — path to peak-frame cortex PNG

### 3. Generate narrations with Gemma 4
After analysis, generate audience-appropriate narrations using the
`jemma-media-pipeline` skill or directly via Ollama:

```bash
ollama run gemma4:e4b-it-q8_0 "$(cat references/tier6_researcher_prompt.txt)"
```

## Output files

All outputs are written to `outputs/`:
- `brain_peak.png` — surface map at peak TR (both hemispheres)
- `preds.npy` — raw predictions `(T, 20484)` float32
- `roi_schaefer400.parquet` — Schaefer-400 ROI time series
- `report.json` — full structured analysis

## Multi-atlas analysis

The `BrainAnalysis.gemma_context()` method generates a rich text block for
injection into Gemma 4 prompts, covering:
- Yeo-7 network ranking with laterality index (LH vs RH dominance)
- Top Schaefer-400 ROIs (functional connectivity parcellation)
- Top Harvard-Oxford ROIs (anatomical, including subcortical)
- Top Jülich regions (Brodmann area mapping)
- Temporal dynamics: rise time, half-max duration, decay slope

## Edge cases

- **Audio-only input** (.wav, .mp3, .flac): V-JEPA2 is skipped; wav2vec-BERT runs
- **Text-only input** (.txt): Only Llama-3.2-3B runs (~10 s). Good for quick previews.
- **Silent video**: Audio features fall back to zero; visual + text encoders still run
- **No GPU**: Will raise `AssertionError: CUDA GPU required` — GPU is mandatory

## Common errors

| Error | Fix |
|-------|-----|
| `FileNotFoundError: tribev2_weights/best.ckpt` | Run `python check_setup.py` to verify weights path |
| `AssertionError: CUDA GPU required` | CUDA must be available; check `nvidia-smi` |
| `shape mismatch` in atlas projection | Usually a nilearn version issue; `pip install nilearn==0.13.1` |
| Ollama connection error | Start Ollama: `ollama serve` |

See [references/REFERENCE.md](references/REFERENCE.md) for the full model card.
