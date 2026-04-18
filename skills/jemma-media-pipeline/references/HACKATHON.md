# Gemma 4 Good Hackathon — Submission Reference

## Competition details

| Field | Value |
|-------|-------|
| Organiser | Kaggle + Google DeepMind |
| Prize pool | $200,000 USD |
| Deadline | **May 18, 2026** |
| Submission URL | https://www.kaggle.com/competitions/gemma-4-good-hackathon |
| Focus areas | Health & Sciences · Global Resilience · Future of Education · Digital Equity · Safety & Trust |

## Judging criteria

| Criterion | Weight |
|-----------|--------|
| Innovation | 30% |
| Impact Potential | 30% |
| Technical Execution | 25% |
| Accessibility | 15% |

## Required submission materials

1. ✅ **Public code repository** (this repo, public GitHub)
2. ✅ **Public project write-up** (Kaggle notebook or external blog post)
3. ✅ **Public demo or demo files** (`/jemma-demo` Discord command + video walkthrough)
4. ✅ **Video presentation** (~3 min screen recording showing live demo)
5. ✅ **Cover image** (cortex PNG from a real demo run)

## Gemma 4 usage in this project

**Required**: Submissions must use at least one Gemma 4 model.

This project uses `gemma4:e4b-it-q8_0` (quantised Gemma 4 E4B) via Ollama
as the **only LLM** in the pipeline, for two distinct purposes:

### 1. Multimodal scene understanding (Stage A)
- Gemma 4 receives 4 evenly-spaced keyframes (512-px JPEG)
- Returns JSON: `content_type`, `subject`, `setting`, `action`, `mood`, `modality`, `description`
- Drives the stimulus label used throughout the analysis

### 2. Adaptive audience narration (Stages B + post-C)
- Stage B: Quick text-only narration from Gemma's own scene description (~220 tokens)
- Post Stage C: 7 expertise-tier narrations from the full `BrainAnalysis.gemma_context()` block:
  - Tier 0 (120 tokens) — toddler
  - Tier 1 (180 tokens) — general adult
  - Tier 2 (260 tokens) — curious adult ← standard Discord output
  - Tier 3 (300 tokens) — high school student
  - Tier 4 (340 tokens) — college adult
  - Tier 5 (420 tokens) — clinician ← standard Discord output
  - Tier 6 (520 tokens) — neuroscience researcher ← standard Discord output

TRIBE v2 is not an LLM — it is a domain-specific neural encoding model
(visual + audio + text → cortical BOLD). It is used as a scientific analysis
tool, not as a reasoning or generation model.

## Why this project fits the hackathon

### Health & Sciences (30% weight match)
- Predicts real cortical brain responses to any media stimulus
- Multi-atlas analysis (Schaefer-400, Harvard-Oxford, Jülich) with anatomical interpretation
- Temporal dynamics: rise time, half-max duration, decay slope
- Clinician-tier narration names specific networks and caveats correctly

### Future of Education (strong fit)
- The 7-tier system adapts the **same brain data** for any audience:
  - A 3-year-old gets a one-sentence story
  - A PhD gets specific Schaefer-400 ROI labels + mean |z| values
- Educators can run the bot in a lecture and get instant, level-appropriate explanations
- No internet required after initial setup

### Digital Equity & Inclusivity
- Fully offline: no API costs after setup
- Runs on consumer hardware (quantised Gemma 4 E4B uses ~6 GB VRAM)
- Discord bot interface: accessible to anyone in the server
- 7 languages of explanation reduce expertise barriers

### Technical innovation
- First public integration of TRIBE v2 brain encoding + Gemma 4 multimodal vision + adaptive narration
- Multi-atlas BrainAnalysis: simultaneous Schaefer-400 + Harvard-Oxford + Jülich projection
- Auto-trim safety: clips over 50 s are automatically trimmed to model limits
- APScheduler health monitoring: GPU temp alerts, worker liveness checks

## Third-party models / tools

| Component | Role | License |
|-----------|------|---------|
| `gemma4:e4b-it-q8_0` | PRIMARY: vision + narration LLM | Gemma Terms of Use |
| TRIBE v2 | Brain encoding (non-LLM) | CC-BY-NC 4.0 |
| Ollama | Local LLM serving | MIT |
| discord.py | Discord bot framework | MIT |
| nilearn | Neuroimaging atlas tools | BSD-3 |
| APScheduler | Cron health monitoring | MIT |
| yt-dlp | Demo asset download | Unlicense |
| ffmpeg | Media processing | LGPL/GPL |

Using supplementary open-source tools alongside Gemma 4 is permitted; the
competition requires that **at least one Gemma 4 model** be used, which
this project satisfies as its primary AI component.

## Nous Research Hermes Agent Hackathon

The March 2026 Nous Research Hermes Agent Hackathon has already closed
(187 submissions, $11,750 prize pool).

**Hermes Agent (MIT)** is fully legal to use as an orchestration layer in
any project. Our skills in `skills/` follow the **agentskills.io open
standard** (originally developed by Anthropic), which is compatible with
Hermes Agent, Claude Code, GitHub Copilot, VS Code, Cursor, and 30+ other
agents.

The `skills/tribe-brain-analysis/` and `skills/jemma-media-pipeline/` skill
packages can be loaded by any agentskills.io-compatible agent, including
Hermes Agent.

## Nous Research RL Environments Hackathon (Atropos)

This is a **separate**, in-person-only hackathon in San Francisco (May 18,
2026) focused on building reinforcement-learning environments using the
Atropos framework. It is not compatible with the Jemma project's goals.

However, the TRIBE v2 brain-response predictions could theoretically be used
as **reward signals** in an RL environment (e.g., optimise a video stimulus
to maximise predicted visual cortex engagement). This would be a future
research direction.
