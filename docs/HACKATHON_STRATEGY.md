# Gemma for Good — Hackathon Strategy (May 18, 2026 submission)

> Scope: reassess what's been built, define what wins, and plan the path from "local brain-narration adapter" → "cloud-served Gemma-4-31B multi-audience explainer that judges will actually click through."
> **Deadline:** 2026-05-18 · **Today:** 2026-04-20 · **Days left:** ~28

---

## 1. Reassessment — where we are as of 2026-04-20

### What's validated
- **TRIBE v2 pipeline** runs end-to-end on RTX 5090: video/audio/text → predicted BOLD → ROI extraction → Gemma narration → 3D brain viewer. Wall clock 4–7 min per clip.
- **Discord bot + webapp** live on localhost with RBAC, queue, rate limiting, audience-tier narration.
- **Brain-narration LoRA v2 (Gemma-3-27B, r=32, α=64, 1000 rows)** — n=30 eval: 6/7 structural-marker claims significant at p<0.05, four at p<10⁻⁴. Adapter *did* learn the narration template.
- **Brain-narration LoRA v3 (Gemma-3-27B, r=64, α=128, 2189 rows)** — zero metrics separate v3 from v2. **Plateau confirmed.** Scaling rows + rank on the same synthetic corpus gives no further signal.
- **Gemma-4-31B brain LoRAs (r32 + r64)** — trained on the 2189-row corpus; benchmarks running now. Adapter-load bug (`Gemma4ClippableLinear`) fixed via `exclude_modules` regex patch.
- **Genuine benchmark harness** — lm-evaluation-harness 0.4.11 wired to our QLoRA variants. Gemma-3 results in: narration LoRA costs ~2–4 pp across arc_challenge / gsm8k / openbookqa / piqa / truthfulqa_mc1 (i.e., the adapter is narrow-band, doesn't cripple general reasoning).

### What's broken / deferred
- **Gemma-4 three-way custom eval** — llama-server returned empty completions for all 30 prompts (chat template issue; GGUF carries a complex Jinja tool-calling template that our llama.cpp build may not parse correctly).
- **No cloud deployment yet** — GCP scripts scaffolded but not running; Cloudflare Tunnel not live.
- **No Kaggle dataset integration** — training corpus is 100% synthetic self-generated narration pairs. Plateau is symptom of that.
- **TRIBE v2 is CC-BY-NC 4.0** — commercial path still gated on Gallant Lab licensing. *Research / educational framing is fine for the hackathon.*

### Diagnosis: why the plateau matters
Same corpus + same template + more rows + bigger rank → **zero improvement**. The adapter is saturated on the *structure* of brain-narration, and the synthetic data doesn't carry enough distributional diversity to reward more capacity. The model learned the template. It didn't learn anything else. More of the same input can't teach it anything else either.

**This is the pivot signal.** Scale the *kinds* of supervision, not the *amount* of the same supervision.

---

## 2. What actually wins "Gemma for Good"

The prize criteria reward: (a) clear social-good narrative, (b) novel/creative Gemma use, (c) a demo a judge can try in <60 s, (d) technical execution the judges can verify.

### Narrative we're going to tell
> **"Gemma explains any stimulus across three audience levels — student, public, expert — grounded in predicted brain response. Runs locally-first, so a neurodivergent teenager in rural Indiana can understand what their senses are doing without handing anything to a cloud API."**

The beat the judges hear:
1. Brain-prediction (TRIBE v2) is visceral and surprising — the *hook*.
2. Multi-tier narration means the same model reaches a child, a journalist, and a clinician without three separate prompts — the *craft*.
3. Local-first Gemma = **privacy and accessibility** = the *social good*.
4. Cloud overflow is the *scale*. The RTX 5090 demos it; Cloud Run handles the rest.

### What's missing to earn that narrative
- A training corpus that teaches Gemma to **actually hit three audience tiers** on *diverse* topics, not just brain narration.
- Kaggle-sourced datasets that broaden beyond neuroscience into plain-language health, education, accessibility.
- A public deployment a judge can open. `brain.redteamkitchen.com` with Cloudflare Tunnel → 5090 → Gemma-4-31B narration.

---

## 3. Pivot: from single-domain to curriculum-trained

### Old approach (v2, v3)
One task, one template, synthetic:
```
Stimulus: <text> + ROIs + peak time  →  3–5 sentence neuroscience narration
```
→ saturates.

### New approach (proposed v4)
Mixed curriculum, three supervision signals braided into one QLoRA:

| Signal | Source | What it teaches | Example output tier |
|---|---|---|---|
| **A. Brain narration** | Existing 2189-row synthetic corpus | Structural marker template + ROI grounding | Expert |
| **B. Tiered scientific explanation** | Kaggle plain-language summary datasets (CORD-19 PLS, Plaba, arXiv abstract-to-plain) | Audience-depth control | Public / Student |
| **C. Medical Q&A** | MedQuAD, HealthQA | Grounded, non-diagnostic explanation style | Public |
| **D. Accessibility-simplified text** | Simple-Wikipedia pairs, OneStopEnglish | Readability control (flesch-kincaid target per tier) | Student |

Dataset target: ~6–8k mixed rows, curated — **not** 50k scraped rows. The plateau lesson was "more rows of the same doesn't help." Diversity, not volume.

---

## 4. Kaggle dataset shortlist (commercial-safe where possible)

Every one of these has a verifiable license. The hackathon submission stays research-framed, but favouring commercially-usable datasets keeps the option open to port the corpus into a future commercial variant.

| Dataset | Slug | Size | License | Use for |
|---|---|---|---|---|
| **MedQuAD** | `jpmiller/layoutlm` (unofficial) or HF mirror | 47k Q&A | CC-BY-SA | Signal C — medical Q&A |
| **Plaba (Plain Language Adaptation of Biomedical Abstracts)** | NLM-hosted; Kaggle mirrors | ~750 pairs | Public domain | Signal B — expert → public |
| **CORD-19 PLS subset** | `allen-institute-for-ai/CORD-19-research-challenge` | ~400k papers, ~20k with PLS | CC-BY-NC + varies | Signal B — filter PLS-paired only |
| **Simple Wikipedia + English Wikipedia paired** | `kkhandekar/wikipedia-for-question-answering` / HF `wiki_auto` | ~200k sent pairs | CC-BY-SA | Signal D — readability simplification |
| **OneStopEnglish corpus** | available on HF, mirrored on Kaggle | ~400 three-tier articles | CC-BY-SA-NC | Signal D — three-tier by design (matches our student/public/expert) |
| **PubMedQA** | HF `pubmed_qa` + Kaggle mirrors | 1k expert + 211k unlabeled | MIT | Signal C — reasoning over medical abstracts |
| **ADReSS / DementiaBank transcripts** | restricted (TalkBank) — skip for hackathon | — | research-only | (later) accessibility — aphasia narration |
| **SciQ** | `allenai/sciq` on HF | 13.7k science MCQ | CC-BY-NC-3.0 | Signal B — student-level science Q&A |
| **ELI5 (cleaned)** | HF `eli5_category` (research fork) | ~270k explanations | cc-by-nc-sa (reddit terms; research-only) | Signal B — "explain like I'm 5" style |

**Hard skips** (license flagged in `docs/DATASET_LEGAL.md`):
- Anything scraped from a ToS'd portal (JSTOR, Purdue licensed DBs, etc.)
- Anything CC-NC if we ever think about commercial — keep commercial-candidate v5 separate from hackathon v4
- Reddit-sourced ELI5 full text — use cleaned research-framed mirrors only

### Acquisition plan
```bash
# One-time setup (human action)
#   1. kaggle.com → Account → Create New API Token → download kaggle.json
#   2. Move to C:/Users/soumi/.kaggle/kaggle.json  (chmod 600 on *nix)
#   3. Verify: kaggle datasets list --max-size 5 2>&1 | head -5

# Then programmatic pulls via scripts/kaggle_pull_corpus.py (to be written):
#   - download, verify license manifest, write dataset cards to docs/datasets/
#   - land raw files at D:/research/corpora/kaggle/<slug>/
#   - produce a normalized jsonl at D:/research/datasets/curriculum_v4_<ts>.jsonl
```

---

## 5. Training plan — Gemma-4-31B v4 "curriculum-brain"

### Config
- Base: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- Method: QLoRA r=64, α=128, dropout=0 (matches v3 config — the rank is not the bottleneck)
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Exclude: `.*\.(vision_tower|audio_tower)\..*` (pre-baked into adapter_config.json)
- Seq len: 2048 · batch 1 · grad_accum 8 · LR 2e-4 (was 1e-4 for v3 — try slightly hotter with more diverse data)
- Epochs: 2 · packing: enabled · BF16
- Expected VRAM: ~26–28 GB · Expected wall clock: 3–5 h on the 5090 for ~6k rows × 2 epochs

### Evaluation (three-layer, required before calling it a win)
1. **Tier-control eval (new)** — 50 held-out stimuli, force each of three tiers, measure Flesch-Kincaid grade + content coverage. Target: ≥2 FK-grade separation between tiers, same fact coverage.
2. **Existing brain-narration custom eval** — 30 held-out Gemma-3 picks re-run so we see whether v4 retains the v2-level structural marker performance.
3. **Genuine benchmarks** — same lm-eval suite (arc_challenge, gsm8k, piqa, openbookqa, truthfulqa_mc1). Regression budget: ≤4 pp vs base (matches what v2 cost). If v4 craters further we roll back.

### Iteration strategy
- Day 1: dataset assembly + 1 smoke run on E4B (fast, ~30 min) to catch any formatting bugs.
- Day 2: first 31B curriculum run.
- Day 3: read evals, adjust mixture weights if any signal is underfit.
- Day 4: second 31B run if needed.

**Karpathy-style autoresearch** (the cloned `scripts/third_party/autoresearch/` pattern) — reserve for *after* the curriculum baseline works. Use it to search LR / packing / mixture ratios at the E4B scale, then port winners to 31B.

---

## 6. Cloud deployment — the "judges can click it" path

### Architecture
```
 judge browser
      │
      ▼
brain.redteamkitchen.com  (Cloudflare DNS + Tunnel)
      │
      ├──▶ RTX 5090 @ home  (live demo, full TRIBE v2 + Gemma-4-31B narration)
      │    localhost:8765 FastAPI · localhost:5173 Vite
      │
      └──▶ GCP Cloud Run   (overflow, cold-start fallback)
           Dockerfile in gcp/Dockerfile.server
           serves pre-computed cached results when 5090 is offline
           gemma4:e4b-it-q8_0 for quick narration (no TRIBE v2 on Cloud Run)
```

### What to deploy
1. **Cloudflare Tunnel live** — domain transfer from Squarespace *or* Cloudflare-for-SaaS with Squarespace DNS until transfer clears.
2. **Webapp + FastAPI hardened** — rate limit, simple auth token for judge access, pre-computed gallery of "cool examples" so judges see results instantly.
3. **GCS-backed result cache** — every finished pipeline's JSON + mesh texture pushed to `gs://rtk-prod-2026-results/` so the Vite frontend can render offline-cached examples.
4. **Cloud Run cold-start backup** — minimal FastAPI that serves the cached gallery; triggers Cloud Tasks to queue jobs for when the 5090 wakes up.

### What NOT to deploy
- **Don't** put the 31B fine-tune on Cloud Run. 31B @ Q4_K_M = 19 GB; Cloud Run has no GPUs by default. Cloud Run GPU is available but expensive and rate-limited.
- **Don't** put TRIBE v2 in the cloud. CC-BY-NC; weights stay on the 5090.

---

## 7. Sprint timeline to 2026-05-18

| Week | Dates | Deliverables |
|---|---|---|
| **W1: data** | Apr 20–26 | Kaggle CLI + credentials · 4 datasets downloaded + license-verified · curriculum jsonl assembled (~6k rows) · E4B smoke fine-tune green |
| **W2: train** | Apr 27–May 3 | Gemma-4-31B v4 curriculum run · tier-control eval built · three-layer eval complete · genuine benchmarks regression check |
| **W3: deploy** | May 4–10 | Cloudflare Tunnel live · judge-shareable URL working · cached gallery of 10 cool examples · Cloud Run fallback deployed · monitoring + rate limits · recovery path for when 5090 offline |
| **W4: polish** | May 11–17 | Demo video (≤3 min) · submission writeup · architecture diagram · reproducibility README · social-good framing writeup (accessibility + neurodiversity) · Submit **May 18** |

### Stop criteria per week
- End of W1: if the curriculum jsonl isn't coherent by Apr 26, stop scaling data and ship v3 (Gemma-3) + polish instead.
- End of W2: if v4 regresses >4 pp on benchmarks OR fails tier control, roll back to v2 for the demo.
- End of W3: if Cloudflare Tunnel + Cloud Run can't both be live, ship just the 5090 + a recorded demo video.

### One-job-per-day discipline
- Morning: one concrete, testable deliverable.
- Afternoon: iterate or pivot.
- Evening: commit + update this doc's checklist section.

---

## 8. Immediate next actions (this week, in order)

1. **[you]** Create Kaggle API token at <kaggle.com/settings> → download `kaggle.json` → place at `C:/Users/soumi/.kaggle/kaggle.json`.
2. **[claude]** Write `scripts/kaggle_pull_corpus.py` that downloads the 4 target datasets with license-card output.
3. **[claude]** Write `scripts/build_curriculum_v4.py` — normalizes each source into `{system, prompt, completion, signal, tier}` rows, writes `D:/research/datasets/curriculum_v4_<ts>.jsonl`.
4. **[claude]** Write `scripts/eval_tier_control.py` — 50 stimuli × 3 tiers → Flesch-Kincaid + fact-overlap scoring.
5. **[claude]** Extend `scripts/finetune_gemma4_brain.py` → `scripts/finetune_gemma4_curriculum.py` with a mixture-weight CLI flag so we can reweight signals without changing source jsonls.
6. **[claude]** Smoke-train on E4B first (30 min), validate pipeline, then launch 31B overnight.
7. **[claude]** Cloudflare Tunnel wiring + a `scripts/deploy_cloud_run.sh` mirror of what's in `gcp/run-inference.sh`.

---

*Last updated: 2026-04-20. This doc is a living plan — each completed item gets its line checked off in the relevant week's row. If priorities shift, edit here before editing code.*
