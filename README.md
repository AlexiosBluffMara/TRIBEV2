# TRIBE v2 on RTX 5090 — ISU Research Partnership Notebook

Runs Meta's [TRIBE v2](https://huggingface.co/facebook/tribev2) brain-activity foundation model locally on an RTX 5090 (Blackwell, sm_120), produces a slide-ready cortex visualization and streaming ROI panel, and frames a joint research program with the **Follmann + Bhattacharya** group at Illinois State University (NIH R15 AREA / NSF CRCNS fit).

The single deliverable is [`tribe_v2_5090_ISU_demo.ipynb`](tribe_v2_5090_ISU_demo.ipynb). It runs end-to-end and bakes its outputs into the notebook so the story is readable without re-executing.

---

## What the notebook produces

| Artifact | What it is | Size |
|---|---|---|
| `outputs/brain_peak.png` | Inflated cortex (LH+RH) with predicted BOLD at peak stimulus moment | ~209 KB |
| `outputs/tribev2_stream.mp4` | Streaming panel — stimulus text scrolling + top-12 Schaefer-400 ROI bars + mean-activity time-series | ~110 KB |
| `outputs/brain_3d.html` | Interactive plotly cortical surface, draggable | ~5.5 MB |
| `outputs/roi_timeseries.parquet` | 400 Schaefer-400 ROI × T time-series | ~345 KB |
| `outputs/stimulus.txt` | The narration used as input | text |

---

## Key facts (for the presentation)

- **Not 70,000 voxels** — TRIBE v2 predicts on the **fsaverage5 cortical surface = 20,484 vertices** (10,242/hemisphere). "70×" is a resolution ratio vs v1 (Algonauts 2025 winner).
- **Training data**: 25 subjects pooled from Algonauts2025 (4) + Lahner2024 BOLD Moments (10) + Lebel2023 Huth narrative (8) + Wen2017 video (3). Not CNeuroMod.
- **Inputs**: video + audio + text in one pass (V-JEPA2-ViT-g, w2v-BERT-2.0, Llama-3.2-3B as frozen feature stacks).
- **Head**: 8-layer transformer (hidden=1152, 8 heads), Conv1d to 20,484-vertex output, 2 Hz, −5 s hemodynamic-lag offset.
- **License**: CC-BY-NC 4.0 — grant-funded academic use is fine.
- **VRAM**: ≈ 15–18 GB at fp16, so the 32 GB 5090 has plenty of headroom.

---

## Setup

```bash
python -m venv .venv
# Windows
source .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# 1. Blackwell-capable torch (5090 needs cu128)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 2. Weights (709 MB)
huggingface-cli login   # needs meta-llama/Llama-3.2-3B access too
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/tribev2', local_dir='tribev2_weights', local_dir_use_symlinks=False)"

# 3. Source
git clone https://github.com/facebookresearch/tribev2.git tribev2_src
pip install --no-deps -e tribev2_src

# 4. Remaining deps
pip install exca neuralset==0.0.2 neuraltrain==0.0.2 x_transformers==1.27.20 \
            einops pyyaml "moviepy>=2.2.1" huggingface_hub gtts langdetect spacy \
            soundfile Levenshtein julius transformers nibabel nilearn matplotlib \
            seaborn plotly ipywidgets colorcet scipy scikit-image pandas tqdm pyarrow \
            jupyter nbconvert ipykernel

# 5. Register kernel and run
python -m ipykernel install --user --name tribev2 --display-name "Python 3 (tribev2)"
jupyter notebook tribe_v2_5090_ISU_demo.ipynb
```

### Gated prerequisite

Meta requires access approval for `meta-llama/Llama-3.2-3B` (used as the text feature extractor). Request at https://huggingface.co/meta-llama/Llama-3.2-3B — approval is usually <5 min.

---

## Windows patches applied

Running on Windows 11 + Python 3.14 + torch 2.11 surfaced four issues. Each is documented in the notebook and reproduced here so a second machine can apply them:

1. **`pathlib.PosixPath` in `config.yaml`** — Meta's `config.yaml` contains a YAML-pickled `PosixPath`. Alias it to `WindowsPath` before any yaml load:
   ```python
   import sys, pathlib
   if sys.platform == 'win32':
       pathlib.PosixPath = pathlib.WindowsPath
   ```
2. **CPU-only torch auto-installed by a transitive dep** — reinstall the cu128 wheel afterwards to restore sm_120 support.
3. **WhisperX + uvx isolation** — `tribev2/eventstransforms.py` hard-codes `compute_type = "float16"`, but `uvx whisperx` runs in its own CPU-only env so float16 is unsupported. Patch to `device = "cpu"` + `compute_type = "int8"`. ~30 s for a short narration, acceptable for the demo.
4. **nilearn API change** — `vol_to_surf(..., interpolation='nearest')` → `'nearest_most_frequent'`.

---

## Research program (section 6 of the notebook)

Three papers, aligned to existing Follmann / Rosa / Stein work and Bhattacharya's digital-twin expertise:

1. Cross-modal transfer: fMRI-pretrained TRIBE v2 → EEG-based seizure precursor detection.
2. Cortical engagement index for remote learning video as a media-quality metric.
3. Dynamical-systems validation: do TRIBE v2 predictions obey the Follmann-Rosa tonic-to-bursting laws?

Grant fit, dataset list, citation support (Mayer, Paivio, Moreno, etc.) and meeting talking points live in the notebook.

---

## Upstream links

- Code: https://github.com/facebookresearch/tribev2
- Weights: https://huggingface.co/facebook/tribev2
- Blog: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
- Live demo: https://aidemos.atmeta.com/tribev2
- Meta Colab: https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb
