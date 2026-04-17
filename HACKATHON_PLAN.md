# MindScope — TRIBE v2 × Gemma 4 E4B for the Gemma 4 Good Hackathon

**Working name:** MindScope (pick final brand in week 1 — shortlist: *MindScope*, *CortexCast*, *NeuroLens*, *InsideYourHead*)
**Deadline:** 2026-05-18 (hackathon close) — **31 days from today (2026-04-17)**
**Prize pool:** $200k across 5 tracks
**Primary track:** Health & Sciences · **Secondary tracks we credibly claim:** Digital Equity, Education, Safety

---

## 1. The one-sentence pitch

MindScope is a production-grade web app that takes any short video+audio+narration clip, runs it through Meta's **TRIBE v2** brain foundation model to predict what the average human cortex would do, and uses **Gemma 4 E4B** — on-device, multimodal — to narrate, explain, and converse about the predicted brain response in plain language for clinicians, researchers, neurodivergent users, and students.

"Upload a cat video. See what your visual cortex, auditory cortex, and language areas would do. Ask Gemma why."

---

## 2. Why this is a strong Gemma 4 Good submission

The hackathon weights ≈ **Innovation 30 / Impact 30 / Technical Execution 25 / Accessibility & offline deployability 15**, and demands "function in low-bandwidth, limited-compute, offline environments." MindScope hits all of them — and if there's an **official Ollama category**, we hit the bullseye:

> **Ollama category advantage:** We use **Ollama** as the primary serving runtime for Gemma 4 E4B. Judges can clone the repo, `docker compose up`, and run the full stack (brain prediction + multimodal LLM narration) on a CPU MacBook or a $30 Raspberry Pi. No GPU required for the demo. This embodies the hackathon's spirit: "AI that runs anywhere."

| Dimension | How we hit it |
|---|---|
| Innovation | First public web app coupling a brain-response foundation model with an on-device multimodal LLM. Nobody is doing this. |
| Impact | Touches real clinical and research workflows (Carle, ISU's Follmann/Bhattacharya group, NIH/NSF pipelines). |
| Technical execution | Real TRIBE v2 inference on an RTX 5090, real Gemma 4 E4B served via Ollama, real websocket streaming, real 3D cortex. Not a mockup. |
| Offline / edge | Gemma 4 E4B runs in **3 GB RAM** via `ggml-org/gemma-4-E4B-it-GGUF` and via **transformers.js WebGPU** in the browser — the explanation layer runs on the clinician's laptop with zero data leaving the hospital. This is the crown jewel for Digital Equity + HIPAA. |

### Multi-track alignment

| Track | Our story | Concrete feature demonstrating it |
|---|---|---|
| **Health & Sciences** (primary) | Brain-response prediction as a low-cost proxy for fMRI-based content testing (seizure triggers, sensory overload, anesthesia candidates). | Upload clip → predicted cortex map + Gemma explanation → clinician can flag regions of concern. |
| **Digital Equity** | Accessibility layer for neurodivergent users: flag content that over-drives sensory cortex (autism spectrum, migraine, photosensitive epilepsy). On-device Gemma means even offline/rural clinics can use it. | "Neurodivergent content review" mode: green/yellow/red overlay on video timeline. |
| **Education** | ISU partnership: intro neuroscience course using live cortex prediction. Paired with the notebook's Paper 2 (cortical engagement index for remote-learning video). | "Classroom mode" — instructor plays a lecture clip, class sees predicted engagement curve in real time. |
| **Safety** | Content safety for mass media: does a trailer over-drive amygdala-adjacent regions? Photosensitive-seizure screening via Follmann-Rosa tonic/bursting test. | "Content-safety report" mode — PDF export flagging risky seconds in the clip. |
| **Global Resilience** | Weakest fit; skip unless we have slack at week 3. |

**We should submit to Health & Sciences as primary and tag Digital Equity and Education as secondary.** The rules allow one primary submission, and multi-track reach strengthens the write-up even when the prize pool is per-track.

---

## 3. The unified concept

Three layers, each independently demo-able:

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Gemma 4 E4B (the narrator / conversational interface)     │
│  • Ingests the raw video+audio DIRECTLY (its own multimodal pass)    │
│  • Ingests TRIBE v2's 20,484-vertex prediction as structured context │
│  • Generates: per-ROI narration, accessibility descriptions, chat Q&A│
│  • Runs on-device (WebGPU browser OR GGUF laptop) OR on Ollama server│
└─────────────────────────────────────────────────────────────────────┘
                 ▲                                    ▲
                 │  predicted BOLD (20,484 × T)       │  raw media
                 │                                    │
┌────────────────┴───────────────┐        ┌──────────┴────────────────┐
│  LAYER 2 — TRIBE v2 head       │        │  LAYER 1 — feature stack  │
│  8-layer transformer           │◄───────│  V-JEPA2 + Wav2Vec-BERT + │
│  hidden=1152, 2 Hz, fsavg5     │        │  Llama-3.2-3B (frozen)    │
│  Runs on RTX 5090 locally      │        │  (future: Gemma variant)  │
└────────────────────────────────┘        └───────────────────────────┘
```

Key design decision: **Gemma is additive, not substitutive, for the MVP.** The trained TRIBE v2 head is locked to Llama-3.2-3B text embeddings. Swapping in Gemma without retraining the projector will destroy prediction quality. So for the hackathon MVP we keep TRIBE v2 exactly as Meta released it, and add Gemma as Layer 3. We pitch the Gemma-backbone swap as the stretch goal + the paper's future work.

---

## 4. Can we replace TRIBE v2's Llama with Gemma while keeping the training?

**Short answer:** Not drop-in. But there are three increasingly expensive paths, and one of them is genuinely in reach for this hackathon.

TRIBE v2's text projector is a learned MLP whose input is Llama-3.2-3B's hidden states (3072-dim, 6 layer checkpoints = 18,432-dim concatenated). Gemma 4 E4B has a different hidden size, different per-layer statistics, and different tokenization. The learned head expects Llama's feature geometry.

| Path | What you do | Training cost | Quality hit | Hackathon viable? |
|---|---|---|---|---|
| **A. Feature-space distillation** | Train a small MLP `g: Gemma_features → Llama-shaped_features` on paired text. ~10M token corpus. Freeze TRIBE v2 head. | ~4–8 GPU-hours on an H100 | 3–8% Pearson drop (estimated from analogous backbone-swap work) | **YES** — week 3 stretch goal. |
| **B. Projector retraining** | Replace TRIBE v2's text projector with a new one sized for Gemma, retrain only that MLP on the original 25-subject fMRI corpus, freeze everything else (head, audio/video projectors, Gemma backbone). | ~40–80 GPU-hours on 1×H100 | 1–3% Pearson drop | **MAYBE** — week 4, only if we can get fMRI training data and H100 time. |
| **C. Full head retrain** | Gemma + V-JEPA2 + Wav2Vec-BERT → retrain the 8-layer head from scratch. | 1000+ GPU-hours | None; potentially better | **NO** — this is the *paper*, not the hackathon. |

**Plan:** MVP ships with Path Zero (TRIBE v2 untouched, Gemma as Layer 3 narrator). Stretch goal is Path A (feature-space distillation) which we write up as a preliminary result. Path B is scoped but deferred to the post-hackathon research program with ISU/Carle.

### Public fMRI datasets for retraining (the paper's data appendix)

| Dataset | Subjects | Hours BOLD | Modalities | Notes |
|---|---|---|---|---|
| Algonauts 2025 | 4 | ~80 | video+audio+text | Already in TRIBE v2 train set |
| Lahner 2024 BOLD Moments | 10 | ~40 | short clips | Already in TRIBE v2 train set |
| Lebel 2023 | 8 | ~20 | narrative audio | Already in TRIBE v2 train set |
| Wen 2017 | 3 | ~10 | movies | Already in TRIBE v2 train set |
| **Natural Scenes Dataset (NSD)** | 8 | ~30 (images) | images | New; enormous. |
| **CNeuroMod** | 6 | ~500 | deep multi-movie | New; gold-standard depth. |
| **StudyForrest** | 20 | ~40 | Forrest Gump | New; open. |
| **HCP 7T movie-watching** | 184 | ~400 | movies | New; population scale. |

**Total retraining pool available: ~1000 hours of BOLD across 240+ subjects**, all under permissive research licenses. This is enough for Path C if we ever want to do a full Gemma-backbone TRIBE v3.

---

## 5. Tech stack

### Frontend
- **Next.js 15** (app router, React Server Components) on **Vercel** for preview + prod deploys
- **TypeScript** strict, **Tailwind + shadcn/ui**
- **react-three-fiber + drei** for the inflated-cortex 3D viewer; reuse the `brain_3d.html` plotly export as fallback
- **Plotly.js** for ROI time-series; **wavesurfer.js** for audio waveform
- **HLS.js** + `<video>` sync with server-pushed ROI frames over **WebSocket (Soketi/Socket.IO)**
- **transformers.js** WebGPU runtime for the offline Gemma demo (uses `onnx-community/gemma-4-E2B-it-ONNX`; E4B ONNX export if available by launch, else E2B for the browser path and E4B for the server path)

### Backend
- **FastAPI** (Python 3.11) for ML inference APIs — already the language TRIBE v2 speaks
- **Ollama** serving Gemma 4 E4B — two paths:
  - **Local:** `ollama pull google/gemma-4-E4B-it` + `ollama run google/gemma-4-E4B-it` (for dev, edge demo, offline test)
  - **Remote:** Ollama running on a cloud GPU (RunPod, Lambda) for public-facing API with the same REST interface
  - Both speak the same `/api/generate` endpoint, so no code changes to swap local ↔ remote
- **TRIBE v2 worker**: a long-running Python process using `TribeModel.from_pretrained("facebook/tribev2")` on the RTX 5090 — one process, one GPU, queue-driven
- **Celery + Redis** for async jobs (feature extraction, TRIBE inference, Gemma generation)
- **PostgreSQL** (Neon or Supabase) for users, jobs, job results, comments
- **Cloudflare R2** (S3-compatible, no egress fees) for video uploads, NIfTI files, prediction artifacts (`.npy`, `.parquet`, `.mp4`)
- **Clerk** for auth with role-based access (public / researcher / clinician)

### Infra & observability
- **Docker Compose** for local dev, one-box prod, and offline edge deployment
  - `docker compose up` spins up `tribev2-worker`, `ollama`, `postgres`, `redis` locally — judges can run this on their own hardware
- **Fly.io** for web + API tier (close to Ollama GPU node)
- **Ollama cloud node**: RunPod or Lambda Labs H100 (same vRAM budget as before, simpler management)
  - No need for batching orchestration; Ollama handles queue natively
  - Judges can also run Ollama on their own hardware — it's designed for that
- **Terraform** skeleton for future AWS HIPAA migration (Carle) — Ollama's simplicity makes this even easier
- **OpenTelemetry → Grafana Cloud** (free tier) for latency/GPU-util
- **Sentry** for errors
- **GitHub Actions** for CI (lint, type, test, deploy-preview on every PR)

### Data format & standards
- **BIDS** for all incoming fMRI (this is what Carle will speak)
- **NIfTI** for volumes, **GIFTI** for surface data, **fsaverage5** for the shared surface
- **DICOM→NIfTI** via `dcm2niix` at ingestion
- All predictions stored as `(subject, task, TR) → 20484-vertex float16 .npy` + a `.parquet` ROI summary

### Why this stack scales to "production load"
- Stateless web + API tier → horizontally scales trivially behind Cloudflare
- Stateful GPU workers → scale by spinning up more RunPod nodes; job queue drains naturally
- Presigned R2 uploads → browser talks directly to storage, API is never in the data path for large files
- CDN caches all predictions (immutable, content-addressed) → second viewer of the same clip pays no GPU cost
- Carle can run the entire stack inside their VPC via the Terraform module — no data leaves HIPAA perimeter

---

## 6. The MVP (what must ship by May 18)

### User flow
1. **Landing page** — hero with an animated inflated cortex, the tagline, "Try the cat-video demo" button, short explainer of TRIBE v2 + Gemma.
2. **Demo page** — user picks the preset 20 s cat clip (or uploads up to 60 s of video ≤ 100 MB).
3. **Processing screen** — progress streamed via websocket: "Extracting video features (V-JEPA2)… 23%" → "Extracting audio features (Wav2Vec-BERT)… 67%" → "Extracting text features (Llama-3.2-3B)… 81%" → "Predicting BOLD on 20,484 vertices…" → "Generating Gemma narration…"
4. **Results page** — three panes:
   - **Left:** the video playing, synced.
   - **Center:** 3D inflated cortex (react-three-fiber) with BOLD colormap animating at 2 Hz, scrubbable, toggleable hemispheres.
   - **Right:** streaming ROI bar chart (top-12 Schaefer-400 regions) + the Gemma narrator feed ("At t=3.2 s, V1 peaks as the cat's stripes move across the visual field…"). Below: a "Ask Gemma about your brain" chat.
5. **Deep-dive tabs** (below the fold):
   - *Modality attribution* — which of {video, audio, text} drove each ROI? (We derive this from TRIBE v2's modality-dropout behaviour.)
   - *Accessibility report* — neurodivergent content flags; CSV + PDF export.
   - *Download predictions* — `.npy`, `.parquet`, `.html` plotly artifacts.
6. **About / Research** — the ISU partnership, the three-paper program, the Carle integration spec, the Gemma-backbone-swap roadmap.

### The cat-video test input
We prepare `assets/demo/cat_20s.mp4`:
- 20 s, 30 fps, 1920×1080
- **Video track:** one cat across three shots (close-up face, full body walking, playing with string)
- **Audio track A:** real purring recording (~25 Hz fundamental, rich harmonics) mixed at –6 dBFS
- **Audio track B:** a human narrator saying in English "Look at the cat. She's purring because she's content. Watch her tail." mixed at –3 dBFS

TRIBE v2 expects all three modalities. This clip exercises visual cortex (cat imagery), auditory cortex (purring + speech), language areas (English narration through Llama-3.2-3B), and temporal/parietal integration (bound audiovisual object).

### Out of scope for MVP
- Live scanner ingestion (just spec the API, stub the endpoint)
- Carle VPC deployment (ship Terraform as deliverable, not running infra)
- Gemma backbone swap training (ship distillation notebook as stretch)
- User-uploaded NIfTI (show only predicted-vs-predicted for now; ground-truth comparison is stretch)

---

## 7. Site architecture diagram

```
         ┌────────────────────────────────────────────┐
         │            Browser (Next.js)               │
         │  3D cortex · ROI panel · Gemma chat (js)   │
         └──────┬────────────────────────▲────────────┘
                │ WebSocket ROI frames   │ presigned upload
                │                        │
         ┌──────▼────────────────────────┴────────────┐
         │         Next.js API / FastAPI               │
         │   auth · job intake · result delivery       │
         └───┬──────────────┬──────────────┬──────────┘
             │              │              │
     ┌───────▼──────┐  ┌────▼─────┐  ┌─────▼──────┐
     │  Postgres    │  │  Redis   │  │ Cloudflare │
     │  (jobs,      │  │  (queue, │  │    R2      │
     │   users)     │  │   cache) │  │ (media,    │
     └──────────────┘  └────┬─────┘  │  NIfTI,    │
                            │        │   preds)   │
              ┌─────────────┴────┐   └────────────┘
              │                  │
   ┌──────────▼────────┐  ┌──────▼─────────────────┐
   │ TRIBE v2 worker   │  │ Ollama Gemma 4 E4B     │
   │ RTX 5090 / H100   │  │ (local dev or RunPod)  │
   │ (Celery consumer) │  │ REST: /api/generate    │
   └───────────────────┘  └────────────────────────┘
```

---

## 8. Sprint plan (31 days)

### Week 1 — 2026-04-18 → 2026-04-24 (Foundations)
- Day 1-2: Monorepo scaffold (`apps/web`, `apps/api`, `workers/tribe`, `workers/gemma`, `infra/`). Next.js 15 + FastAPI + Docker Compose boot.
- Day 3: Wrap the notebook's inference path into a clean `workers/tribe/serve.py` that consumes a Celery job and emits `.npy` + `.parquet` to R2.
- Day 4: Stand up Ollama with `google/gemma-4-E4B-it`. Test multimodal pass with a still image + audio clip via `/api/generate`.
- Day 5: Websocket ROI streaming prototype (fake data → frontend panel renders).
- Day 6-7: First end-to-end: preset cat clip → TRIBE v2 preds → static 3D cortex render on a page.

### Week 2 — 2026-04-25 → 2026-05-01 (Core UX)
- Day 8-9: react-three-fiber inflated-cortex viewer with scrubbable BOLD colormap (convert fsaverage5 GIFTI → glTF once, serve from R2).
- Day 10: Gemma narration prompt design — we feed Gemma (a) ROI-top-k time series as structured text, (b) the raw video+audio (it's multimodal), (c) a Schaefer-400 atlas key. Stream tokens to UI.
- Day 11: "Ask Gemma about your brain" chat — RAG over the ROI series + a small neuroanatomy corpus.
- Day 12: User uploads + presigned R2 + ingestion pipeline.
- Day 13-14: Polish, error states, loading states, mobile layout. First public preview deploy.

### Week 3 — 2026-05-02 → 2026-05-08 (Differentiators + stretch)
- Day 15: Neurodivergent accessibility mode — sensory overload heuristics over predicted ROIs, red/yellow/green timeline.
- Day 16: Classroom mode — multi-viewer shared playback with live engagement curve.
- Day 17: Content-safety PDF export (the Safety track angle).
- Day 18: **Stretch: Path A distillation notebook** — train `Gemma_features → Llama-features` MLP on 10M-token corpus, plug into TRIBE v2, report Pearson drop.
- Day 19: **Stretch: browser-side Gemma** — transformers.js WebGPU offline mode behind a toggle.
- Day 20: Write the Terraform module + Carle integration spec (not deployed; documented).
- Day 21: Load test with k6 (100 concurrent viewers on cached clip, 5 concurrent uploads).

### Week 4 — 2026-05-09 → 2026-05-15 (Polish, write-up, video)
- Day 22-23: Visual polish, copywriting, landing-page hero animation.
- Day 24: Technical write-up (the required "how Gemma 4 was applied" doc).
- Day 25: Demo video shoot (3 min, covers: cat demo → clinician demo → offline WebGPU demo).
- Day 26-27: Final bug bash, a11y audit, perf audit (Lighthouse ≥ 90 across the board).
- Day 28: Public code repo cleanup, LICENSE, CONTRIBUTING, detailed README.

### Buffer — 2026-05-16 → 2026-05-17
- Submission dry run, backup recording of the demo, fix whatever breaks.

### Submit — 2026-05-18

---

## 9. Scalability, HIPAA, Carle integration

**Real production-load readiness** (the user asked for this explicitly):

- **Ingestion:** presigned multipart R2 uploads; 5 GB max per file; SHA-256 content-addressed so duplicate fMRI volumes dedupe automatically.
- **Processing:** Celery queue with priority tiers (public demo = low, researcher = normal, clinical = high). Each GPU worker advertises available VRAM; scheduler picks a worker that fits.
- **Cost control:** cached prediction reuse (content-addressed); GPU autoscaling with 60 s scale-down; R2 has no egress fees so CDN is free.
- **HIPAA readiness:** we are **not HIPAA-certified for MVP**, and we say so clearly. The Terraform module we ship stands up the same stack inside Carle's own AWS account (BAA-covered region, KMS at rest, TLS in transit, VPC peering). Data never leaves Carle's perimeter. The write-up includes a BAA checklist.
- **De-identification:** `pydicom` + `deid` to strip PHI at ingestion if Carle pushes DICOM.
- **Observability:** OTel traces from browser through API through workers; request-scoped PII redaction in logs.

**Carle / ISU integration plan:**
- Carle contact path: the README already names the Follmann + Bhattacharya group at ISU. Parallel outreach to Beckman Institute (UIUC) since most clinical-research scanners in the region sit there.
- **Week 2 action:** send the scanner-integration spec to Follmann. Ask if ISU's IRB has a data-use template we can borrow.
- **Post-hackathon:** NIH R15 AREA grant to fund the Gemma-backbone TRIBE v3 training. NSF CRCNS for the cross-modal EEG transfer (notebook Paper 1).

---

## 10. The roadmap for a "Gemma-native TRIBE v3"

This is what the hackathon write-up pitches as future work — judges love a credible research horizon.

1. **Phase 0 (done by submission):** Gemma 4 E4B as narration layer. Path-A distillation notebook proving < 10% Pearson drop.
2. **Phase 1 (post-hackathon, 3 months):** Path B — retrain only the Gemma→head text projector on the 25-subject pool using the existing TRIBE v2 training loop. Expected Pearson drop: 1–3%. Deliverable: `facebook/tribev2` → `philanthropytraders/tribe-gemma-v2` on HF.
3. **Phase 2 (6 months, with ISU/Carle):** Path C — full head retrain with Gemma 4 E4B text + new Gemma audio encoder (replacing Wav2Vec-BERT) + V-JEPA2. Expand training to NSD + CNeuroMod + HCP 7T. Target: match or beat TRIBE v2 Pearson on held-out subjects.
4. **Phase 3 (9+ months):** Personalization layer — 10 minutes of movie-watching at Carle to fit a per-patient subject-layer, producing a personalised brain-response model. This is the clinically useful product.

**Compute estimate for Phase 2:** ~1200 H100-hours (~$3k on RunPod at current spot rates, ~$15k on-demand). Fits a small research grant.

---

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| RTX 5090 flakiness under sustained load | Medium | Fall back to RunPod H100 for public demo. 5090 stays for dev. |
| Ollama model registry / network latency for cold pulls | Medium | Pre-cache `google/gemma-4-E4B-it` in Docker build step; offline demo fallback via transformers.js WebGPU. |
| transformers.js WebGPU E4B too slow in browser | High | Ship the offline demo with E2B instead; pitch "E4B server / E2B device" as a feature, not a bug. |
| 31 days too short for the distillation stretch | Medium | Cut Path A before week 3; leave notebook as "executable appendix". |
| Licensing confusion (TRIBE v2 CC-BY-NC, Gemma Apache 2.0, Llama-3.2 custom) | Low | Separate code (Apache 2.0) from model weights (original licenses retained). Non-commercial banner for the TRIBE v2 predictions. The Kaggle submission is academic research, which TRIBE v2's license explicitly permits. |
| Carle / ISU not responsive in hackathon window | High | The Carle/ISU angle is *narrative* for the write-up, not *blocking* for the demo. We ship without either being live. |
| Judges discount us as "just a wrapper" | Medium | Emphasize the Path-A distillation result, the HIPAA Terraform module, the on-device WebGPU Gemma path, and the neurodivergent accessibility mode — four pieces no wrapper has. |

---

## 12. Immediate next actions (what to do tomorrow morning)

1. Decide project name. I recommend **MindScope**.
2. `git checkout -b mindscope-scaffold`, scaffold the monorepo (`pnpm create next-app@latest apps/web --typescript --tailwind --app`, `uv init apps/api`, add `workers/tribe` and `workers/gemma` packages, add `infra/terraform`).
3. Move the notebook's inference path into `workers/tribe/serve.py` behind a Celery task. Keep the notebook as a reproducible research artifact; the web app calls the extracted code.
4. Register for the hackathon on Kaggle (the competition entry — [The Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)).
5. Provision a Cloudflare R2 bucket, a Neon Postgres, a RunPod H100 reservation for the video-shoot week.
6. Open three tracking issues: (Health primary), (Digital Equity secondary — WebGPU Gemma), (Education secondary — Classroom mode).

---

## Sources

- [The Gemma 4 Good Hackathon — Kaggle](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
- [Kaggle and Google DeepMind open Gemma 4 AI hackathon — EdTech Innovation Hub](https://www.edtechinnovationhub.com/news/kaggle-and-google-deepmind-open-gemma-4-hackathon-focused-on-ai-skills-and-real-world-impact)
- [Welcome Gemma 4: Frontier multimodal intelligence on device — HF blog](https://huggingface.co/blog/gemma4)
- [Ollama — Local LLM serving](https://ollama.ai)
- [Gemma models on Ollama](https://ollama.com/search?q=gemma)
- [facebook/tribev2 — HuggingFace](https://huggingface.co/facebook/tribev2)
- [TRIBE v2 source — facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)
