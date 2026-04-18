# TRIBE v2 / Jemma / Red Team Kitchen — Project Context

This file orients Claude Code sessions on this repo. Read before making changes.

## What this project is

JemmaBrain: takes short video/audio/text clips, predicts cortical BOLD response using TRIBE v2 (Gallant Lab, UC Berkeley), narrates the prediction using Gemma across three audience tiers (student / public / expert), and renders it in an interactive 3D brain viewer.

- **Legal entity:** Alexios Bluff Mara LLC (dba Red Team Kitchen)
- **Product domain:** brain.redteamkitchen.com (app) + redteamkitchen.com (marketing)
- **Hackathon target:** Gemma for Good (submit May 18, 2026) + Nous Hermes (secondary)

## Hard invariants — do not break

1. **`duration_trs` must not be overridden in TRIBE model config.** Passing `config_update={"duration_trs": ...}` to `TribeModel.from_pretrained` breaks keep-mask indexing and produces silent garbage output. See `memory/feedback_tribe_config.md`.
2. **All Windows subprocess calls to `ffmpeg` / `nvidia-smi` from asyncio must use `CREATE_NO_WINDOW`**. Otherwise console windows flash on every call and the bot terminal gets destroyed. See `memory/feedback_subprocess.md`.
3. **TRIBE v2 is CC-BY-NC 4.0.** Non-commercial use only. The Red Team Kitchen commercial roadmap (`docs/BUDGET.md` "Revenue Model") must either stay in research/educational framing OR arrange commercial licensing with Gallant Lab before charging for inference output.
4. **Gemma is local-first via Ollama, not Gemini API.** Full sovereignty is the pitch. See `docs/KIMI_VS_GEMMA.md`. Don't introduce Gemini or other external LLM APIs into the hot path.

## Stack at a glance

- **Compute:** Windows 11 · RTX 5090 (32 GB GDDR7, Blackwell sm_120) · 64 GB RAM
- **Python venv:** `C:\Users\soumi\venvs\tribev2` (NOT on D: drive)
- **Inference:** TRIBE v2 BF16 + torch.compile, 4–7 min per full multimodal pipeline
- **LLM:** Gemma 3 27B via Unsloth GGUFs in Ollama — Q8_0 for narration, Q5_K_M for agents
- **Backend:** FastAPI (`webapp/server.py`) on localhost:8765
- **Frontend:** Vite + Three.js (`webapp/`) on localhost:5173
- **Bot:** Discord.py (`bot/`) with RBAC, priority queue, rate limiting, analysis threads
- **Cloud:** GCP project `rtk-prod-2026` — Cloud Run API, Cloud Tasks queue, GCS, Secret Manager
- **Edge:** Cloudflare Tunnel exposes the 5090 as `brain.redteamkitchen.com`
- **Registrar + DNS:** Cloudflare (post-transfer from Squarespace)
- **Email:** Google Workspace on `philanthropytraders.com` + alias domain `redteamkitchen.com`

## Directory map

```
D:/TRIBEV2/
├── bot/                  # Discord bot + core pipeline
│   ├── bot.py            # Discord client, slash commands
│   ├── pipeline.py       # TRIBE v2 inference orchestration
│   ├── gemma.py          # Narration templates per audience tier
│   ├── ollama_client.py  # Gemma via Ollama REST
│   ├── cat_gate.py       # Classification (visual vs non-visual)
│   ├── tiers.py          # Seven-tier narration selection logic
│   ├── visualize.py      # Matplotlib heatmap renders
│   ├── model_manager.py  # Ollama model lifecycle
│   ├── tunnel.py         # Cloudflare Tunnel helpers
│   └── gcs_store.py      # GCS result persistence
├── webapp/               # Vite + Three.js brain viewer
│   ├── server.py         # FastAPI backend (runs alongside bot)
│   ├── src/              # Vite source
│   └── public/           # Static assets
├── gcp/                  # GCP infrastructure
│   ├── init.py           # Project + API + bucket bootstrap
│   ├── Dockerfile.server
│   ├── cloudbuild-server.yaml
│   └── run-inference.sh
├── scripts/              # One-off + dev scripts
│   └── gemma_pulls.sh    # Ollama pull commands
├── docs/                 # Planning + specs
│   ├── BUDGET.md
│   ├── ISU_PROPOSAL.md
│   ├── DATASET_LEGAL.md
│   ├── CONSOLIDATION.md
│   ├── KIMI_VS_GEMMA.md
│   ├── LINKEDIN_DECORATORS.md
│   ├── REGISTRAR_COMPARISON.md
│   ├── GCP_SETUP.md
│   └── FACULTY_SPONSOR_EMAIL.md
├── squarespace/          # Marketing site HTML
├── outputs/              # Local result artifacts (not in git)
└── ROADMAP.md            # Top-level phase plan
```

## Common commands

```bash
# Start bot (runs webapp too)
python -m bot

# Run webapp alone (dev)
cd webapp && npm run dev          # Vite on :5173
python webapp/server.py           # FastAPI on :8765

# Gemma model pulls (after Ollama installed)
bash scripts/gemma_pulls.sh

# Run one pipeline locally for a test video
python -m bot.run_demo <path-to-video>

# GCP auth + project context
gcloud auth login
gcloud config set project rtk-prod-2026

# Deploy API to Cloud Run (production)
gcloud builds submit --config gcp/cloudbuild-server.yaml
```

## Style + conventions

- **No `Co-Authored-By: Claude` in commits.** Ever. Match existing commit style (look at `git log`).
- **No `# Generated by Claude`-style comments in code.** Ever.
- **No unprovoked docstrings or comments.** Only write a comment when the *why* is non-obvious.
- **No error handlers for impossible cases.** Trust internal code; validate only at system boundaries (user input, network).
- **Prefer editing existing files over creating new ones.**

## What lives where (decision tree for new code)

- **New Discord command** → `bot/bot.py` (slash commands) + logic in the relevant `bot/*.py` module
- **New API endpoint** → `webapp/server.py`
- **New Three.js feature** → `webapp/src/`
- **New GCP resource** → `gcp/init.py` (Python) or the shell scripts if it's a one-off
- **New doc** → `docs/` with a consistent name pattern

## Things currently broken or deferred

- **Workspace primary email ambiguity:** plan to keep `soumitlahiri@philanthropytraders.com` as primary. If this changes, `docs/CONSOLIDATION.md` and `bot/config.py` both need updates.
- **Redteamkitchen.com is owned by soumitty@gmail.com on Squarespace.** Transfer to Cloudflare Registrar pending.
- **Pixel 9 Pro Fold app:** post-hackathon (July+).
- **Payment portal / donations / crypto:** post-hackathon, pending lawyer review — LLCs can't accept "donations" as gifts per `docs/CONSOLIDATION.md` flag.
- **Scraping Google/Meta job boards:** do NOT implement. Use official Greenhouse JSON endpoints + LinkedIn email alerts instead.

## Memory system

User has auto-memory at `C:\Users\soumi\.claude\projects\D--TRIBEV2\memory\`. Key existing entries:
- `project_jemma.md` — architecture and working paths
- `feedback_subprocess.md` — Windows subprocess CREATE_NO_WINDOW requirement
- `feedback_tribe_config.md` — duration_trs invariant
- `user_soumit.md` — user profile

New session context goes to memory; this CLAUDE.md is for project facts that apply to anyone working in the repo.

---

*Last updated: April 2026*
