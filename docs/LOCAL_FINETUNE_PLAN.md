# Local Fine-Tune + Research Corpus Plan — Offline on the 5090

## Goal

Build a massive, fully-local, offline-processable research corpus and fine-tune a custom Gemma 3 variant on the RTX 5090. Strictly personal research. **Completely separate from Red Team Kitchen's commercial product** — no shared weights, no shared training data, no shared claims. The commercial product continues to run stock Unsloth Gemma GGUFs with its own separately-sourced training data.

## What this plan is NOT

- **Not** automated scraping of Purdue/ISU/CPL licensed databases, even at low rate. See `docs/DATASET_LEGAL.md` for why that's off-limits. Rate-limiting does not convert a ToS violation into compliance; detection is multi-signal, and the downside (account loss, sponsorship termination, publisher lawsuit, LLC exposure, contaminated weights in any future diligence) is much larger than the upside.
- **Not** a path to using ingested content in the commercial product. If a data source isn't explicitly commercial-use, its derivatives stay on the research side of the fence, permanently.

## Legal corpora that are genuinely massive

Each of these alone is multi-TB. Pick what matches your actual research question; don't hoard.

| Corpus | Size | License | Notes |
|--------|------|---------|-------|
| **FineWeb** (HuggingFaceFW) | ~15T tokens | Apache 2.0 | Best-in-class cleaned Common Crawl derivative |
| **FineWeb-Edu** | 1.3T tokens | Apache 2.0 | Educational-quality subset, great for reasoning |
| **Dolma v1.7** (AllenAI) | ~3T tokens | ODC-BY | Well-documented pretraining corpus |
| **RedPajama-v2** | ~30T tokens | Apache 2.0 (code/metadata) | Massive scale; per-doc licenses vary |
| **The Pile (uncopyrighted)** | ~800B tokens | Permissive subset only | Safe subset of original Pile |
| **PubMed Central OA Commercial** | ~4.5M articles | CC-BY / CC0 varies per article | Filter by `oa_comm` subset only |
| **arXiv bulk S3** | ~2.5M preprints, ~2.7TB | Per-article license, many CC-BY | `s3://arxiv/` requester-pays |
| **Wikipedia dumps** | ~100GB | CC-BY-SA 4.0 | Attribution + share-alike required |
| **Wikidata** | ~100GB | CC0 | Structured facts |
| **Common Crawl** | ~400TB per crawl | Open terms | Filter yourself or use FineWeb instead |
| **OpenWebMath** | 14B tokens | ODC-BY | Math-focused crawl |
| **The Stack v2** | ~900B tokens code | Permissive licenses only | Code pretraining |
| **Project Gutenberg** | ~70k books | US public domain | Pre-1929 literature |
| **LoC / NARA / NIH / NSF data** | varies | US gov public domain | Historical + scientific |

**For fMRI/neuro specifically:** OpenNeuro, HCP, NSD, Algonauts, StudyForrest, NeuroVault. See `docs/DATASET_LEGAL.md`.

## Storage plan

Two NVMe SSDs is the sweet spot:
- **Primary SSD** (4TB, e.g. WD SN850X or Samsung 990 Pro, ~$260–300): Wikipedia + Wikidata + arXiv + PubMed OA + FineWeb-Edu (1.3T subset) + training output. Fits comfortably.
- **Archive SSD** (4–8TB, external USB4 or internal NVMe): full FineWeb, Dolma, The Stack — mounted read-only during training runs.

Don't try to keep all 30T of RedPajama-v2 locally. Stream from HF Datasets with on-the-fly tokenization instead.

## Multimodal Gemma 3 on RTX 5090

Gemma 3 is natively multimodal (vision encoder baked in at 4B / 12B / 27B sizes; the 1B is text-only). Key variants for this box:

| Variant | Format | VRAM (bf16) | VRAM (4bit) | Use case |
|---------|--------|-------------|-------------|----------|
| `google/gemma-3-27b-it` | HF (bf16) | ~54GB | — | Too big for bf16 on 5090 |
| `unsloth/gemma-3-27b-it-bnb-4bit` | HF (4bit) | — | ~18GB | Vision-capable, fits with headroom for LoRA |
| `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` | HF (dynamic 4bit) | — | ~18GB | Best accuracy/speed tradeoff — use this |
| `unsloth/gemma-3-27b-it-GGUF:Q8_0` | GGUF | — | ~29GB | Ollama path for narration (text only, practically) |
| `unsloth/gemma-3-12b-it-bnb-4bit` | HF (4bit) | — | ~9GB | Fast agent loop + vision; room for full LoRA |
| `unsloth/gemma-3-4b-it-bnb-4bit` | HF (4bit) | — | ~3.5GB | Classifier / quality filter / distillation target |

**For the Hermes agent w/ vision:** use `unsloth/gemma-3-12b-it-unsloth-bnb-4bit` through transformers (not Ollama). 12B is enough for tool-use loops, gives you vision, and leaves ~20GB VRAM for KV cache and concurrent workloads.

**For narration (text only):** stay on the GGUF/Ollama path per `docs/KIMI_VS_GEMMA.md`. No reason to switch.

## Fine-tuning on the 5090

Blackwell (sm_120) requires PyTorch 2.7+ and recent CUDA (12.8+). Unsloth added native Blackwell support mid-2025. Verify before committing to a training run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# Expected: 2.7+ and (12, 0) for Blackwell
```

### Framework choice

- **Unsloth** (recommended) — 2× faster, ~50% less VRAM than vanilla HF Trainer, Blackwell supported. One-liner setup, good defaults.
- **Axolotl** — more flexible configs, slower than Unsloth, good for DPO/ORPO pipelines.
- **Torchtune** — PyTorch-native, cleaner code, fewer batteries included.

Default choice: **Unsloth + QLoRA** for 27B, **Unsloth + LoRA** for 12B.

### What fits on 32GB

| Model | Method | Seq len | Batch | VRAM | Notes |
|-------|--------|---------|-------|------|-------|
| Gemma 3 27B | QLoRA (r=16) | 2048 | 1 + grad accum 8 | ~28GB | Tight; disable logging images |
| Gemma 3 27B | QLoRA (r=64) | 4096 | 1 + grad accum 4 | ~30GB | Pushes limits |
| Gemma 3 12B | LoRA (r=64) | 4096 | 2 + grad accum 4 | ~22GB | Comfortable |
| Gemma 3 12B | Full FT | — | — | OOM | Needs multi-GPU |
| Gemma 3 4B | Full FT | 4096 | 2 + grad accum 8 | ~24GB | Viable for small distillation targets |

### Recommended fine-tune targets

1. **`gemma-3-27b-jemma-narration-v1`** — narration voice tuning. Train QLoRA on ~50–100k curated narration pairs (tier 5-6 style from `bot/tiers.py`). ~24–48h run.
2. **`gemma-3-12b-hermes-agent-v1`** — agent-loop / tool-use tuning. Train LoRA on Hermes-style agentic traces + your own pipeline traces. ~12–24h run.
3. **`gemma-3-4b-gatekeeper-v1`** — distilled classifier for visual-vs-non-visual gate (`bot/cat_gate.py`). Full FT on a few thousand labeled examples. ~2h.

Export path: LoRA → merged weights → GGUF → Ollama tag. Unsloth has `save_pretrained_gguf()` that handles this.

### Research-only weights, strictly

Every model trained on research-corpus content gets tagged in its name (`-research-v1`) and never ships to production. The commercial pipeline stays on stock Unsloth GGUFs until a cleanly-licensed training set exists for commercial fine-tuning.

## End-to-end workflow

```
raw legal corpora (HF datasets / S3 / FTP)
    ↓
PDF → markdown  (marker primary, docling fallback, PyMuPDF4LLM for speed)
    ↓
local chunk + dedupe  (datatrove by HF)
    ↓
quality filter  (gemma-3-4b classifier, your own prompts)
    ↓
split:
  ├── RAG index    (LanceDB or Qdrant on SSD)
  └── fine-tune dataset  (HF datasets shards, parquet on SSD)
    ↓
Unsloth QLoRA train  (RTX 5090, 24–48h)
    ↓
merge + GGUF export
    ↓
Ollama tag  (local use only, `jemma-research-27b-v1`)
    ↓
agent loop / narration experiments via Hermes
```

## What to actually do this week

1. Verify Unsloth Blackwell support: `pip install unsloth && python -c "import unsloth; print(unsloth.__version__)"` — must be 2025.10+ for stable sm_120.
2. Download FineWeb-Edu 1.3T subset via HF Datasets streaming API — don't bulk download, stream + tokenize to disk with datatrove.
3. Pull `unsloth/gemma-3-12b-it-unsloth-bnb-4bit` for the agent loop experiments.
4. One test QLoRA run on Gemma 3 12B with 1000 examples just to confirm Blackwell + Unsloth + LoRA pipeline is green end-to-end. Don't commit to the 27B run until 12B works.

## Cost

Entirely local. Electricity for a 48h training run at 5090 full draw (~500W) is ~$3 at Chicago residential rates. That's the entire variable cost. No cloud spend, no API spend, no dataset cost.

## Crystal-clear separation from the commercial product

- Research corpora live at `D:/research/corpora/`.
- Research models live at `D:/research/weights/` and are Ollama-tagged with a `research-` prefix.
- The `bot/` pipeline imports **only** stock Unsloth Gemma tags, never the research tags.
- Research work is logged in a separate journal, not in commit messages on `bot/` code.
- If the LLC ever considers using any of this research work in the product, that triggers a full licensing review of every source used in training. Default is no.

---

*Last updated: April 2026*
