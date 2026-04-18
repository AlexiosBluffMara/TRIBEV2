# JemmaBrain × TRIBE v2 — Product Roadmap & Go-to-Market Strategy

> **Alexios Bluff Mara LLC** · soumitlahiri@philanthropytraders.com  
> **Domain:** redteamkitchen.com (Squarespace) → `brain.redteamkitchen.com` (dynamic app)  
> **Hardware:** Windows 11 · RTX 5090 (32 GB GDDR7, sm_120 Blackwell) · 64 GB RAM  
> **Stack:** TRIBE v2 · Gemma 4 (E4B / 26B MoE / 31B) · Ollama · Three.js · Discord · FastAPI

---

## 1. Vision

Make fMRI-grade predicted brain response data **accessible to three audiences simultaneously**:

| Audience | Mode | Example |
|---|---|---|
| 🎓 High school / patient | Student | "Your focus network lit up during tense moments" |
| 👥 Curious adult / journalist | Public | "The Default Mode Network deactivated at t=2.1s" |
| 🩺 Neurologist / researcher | Expert | "LH DorsAttn peak z=3.2, bilateral SalVentAttn, rise 0.8s" |

**Core differentiator:** TRIBE v2 runs *predicted* BOLD response (no fMRI scanner needed) from any short video/audio/text clip, narrated by Gemma 4 with per-audience depth, displayed in real-time 3D.

---

## 2. Current State (Local-First, April 2026)

### ✅ Working
- TRIBE v2 inference on RTX 5090 (BF16, torch.compile, 4-7 min full multimodal)
- Gemma 4 three-tier narration (E4B / 26B MoE / 31B dense via Ollama)
- Discord bot with RBAC, priority queue, rate limiting, analysis threads
- Three.js brain viewer (PBR shaders, Yeo-7 networks, animated BOLD heatmap)
- FastAPI server + Vite webapp (localhost:8765 / 5173)
- GCS result store (optional), GCP infrastructure scripts

### 🔧 In Progress
- Internet exposure (Cloudflare Tunnel setup)
- Audience mode UI (Student / Public / Expert)
- Central results feed channel
- WebGPU progressive enhancement

### ❌ Not Yet Built
- Public website on `brain.redteamkitchen.com`
- Persistent queue when PC is offline
- GCP hybrid inference (overflow + failover)
- Stripe payment for API access
- Performance benchmark dashboard (captured tok/s, quality scores)

---

## 3. Internet Exposure — Expose the RTX 5090 NOW

### Option A: Cloudflare Tunnel (Recommended — Free, No Port Forwarding)

```powershell
# 1. Install cloudflared on your Windows PC
winget install Cloudflare.cloudflared

# 2. Login to Cloudflare (creates ~/.cloudflared/cert.pem)
cloudflared login

# 3. Create a named tunnel
cloudflared tunnel create jemmabrain
# → saves tunnel credentials to ~/.cloudflared/<UUID>.json

# 4. Create config file: C:\Users\soumi\.cloudflared\config.yml
# tunnel: <your-tunnel-UUID>
# credentials-file: C:\Users\soumi\.cloudflared\<UUID>.json
# ingress:
#   - hostname: brain.redteamkitchen.com
#     service: http://localhost:8765
#   - service: http_status:404

# 5. Add DNS record in Squarespace:
#   Type: CNAME
#   Host: brain
#   Points to: <your-tunnel-UUID>.cfargotunnel.com

# 6. Run the tunnel (or install as Windows service)
cloudflared tunnel run jemmabrain

# As a persistent Windows service:
cloudflared service install
net start cloudflared
```

**Result:** `https://brain.redteamkitchen.com` → tunnels to `localhost:8765` on your RTX 5090.  
- HTTPS automatic (Cloudflare cert)  
- No port forwarding  
- No dynamic DNS  
- Free tier: unlimited bandwidth for tunnels  

### Option B: Tailscale (Private team access only)

```powershell
# Install Tailscale on your PC and any device needing access
winget install Tailscale.Tailscale
# → All devices see each other as if on same LAN
# → Use Tailscale IP (e.g. 100.x.x.x:8765) to access brain viewer
```

**Best for:** Internal team, collaborators, not for public submissions.

### Option C: ZeroTier (Peer-to-peer, free up to 25 devices)

Similar to Tailscale — good for research team scenarios.

### When the PC is Off — Queue Persistence

```
User submits job → Cloud Run receives → Cloud Tasks queues job
      ↓
PC online? → Cloud Tasks polls PC → PC processes → uploads to GCS
PC offline? → job stays in Cloud Tasks (no timeout for 1 hour) → retried when PC comes back
      ↓  (PC down > 1 hour)
Fallback → GCP L4 preemptible VM launched → processes job → uploads to GCS
```

### Windows Auto-Start (Ensure PC Always Serving)

```powershell
# Create a Windows Task Scheduler entry to start Jemma on login:
schtasks /create /tn "JemmaBrain" /tr "C:\TRIBEV2\start_bot.bat" /sc onlogon /ru SYSTEM

# start_bot.bat:
@echo off
cd D:\TRIBEV2
call C:\Users\soumi\AppData\Local\Python\Python312\python.exe -m bot
```

---

## 4. Website Architecture

### Domain Structure

```
redteamkitchen.com          ← Squarespace (marketing, about, team)
brain.redteamkitchen.com    ← Cloudflare Tunnel → RTX 5090 (primary)
                             ← Cloud Run fallback (when PC is off)
api.redteamkitchen.com      ← Cloud Run (job submission API, always-on)
```

### redteamkitchen.com (Squarespace)

Pages:
1. **Home** — hero video showing brain animation, 30-second demo
2. **About** — TRIBE v2 explanation, team, Alexios Bluff Mara LLC
3. **Try It** — link to `brain.redteamkitchen.com` (the live app)
4. **Science** — what TRIBE v2 is, fMRI primer, paper links
5. **For Researchers** — API docs, rate limits, partnership inquiries
6. **Blog** — hackathon progress, case studies, media coverage

### brain.redteamkitchen.com (Vite App)

Three-panel layout:
```
┌─────────────────────────────────────────────┐
│  [🎓 Student]  [👥 Public]  [🩺 Expert]     │  ← Audience toggle
│                                             │
│         3D Brain (Three.js)                 │
│         Rotating, animated BOLD             │
│                                             │
│  [Drag video here to analyze]               │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ Network: Default Mode (active)       │   │
│  │ "Your daydreaming network lit up..." │   │  ← Audience-appropriate text
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 5. Phase 1: Local-First Polish (Weeks 1–4, NOW)

| Task | Status | Est. Time |
|---|---|---|
| Fix brain renderer clipping (two-pass rendering) | ✅ Done | — |
| Add audience mode (Student/Public/Expert toggle) | ✅ Done | — |
| Cloudflare Tunnel setup | 🔧 Ready | 30 min |
| Discord bot RBAC + priority queue | ✅ Done | — |
| Performance metrics capture | 📋 Next | 2 hrs |
| Squarespace landing page | 📋 Next | 4 hrs |

### Performance Metrics to Capture and Display

Create `D:\TRIBEV2\bot\benchmarks.py`:

```python
# Metrics to capture per pipeline run:
METRICS = {
    # Inference
    "tribe_inference_secs":      float,   # Stage C wall time
    "tribe_shape":               tuple,   # (n_trs, n_verts)
    "model_dtype":               str,     # 'bf16'
    "gpu_name":                  str,     # 'NVIDIA GeForce RTX 5090'
    "gpu_vram_allocated_gb":     float,
    "gpu_util_pct":              int,     # from nvidia-smi
    # Gemma narration
    "gemma_model":               str,     # 'gemma4:26b'
    "gemma_tok_per_s":           float,   # from Ollama eval_count/eval_duration
    "gemma_num_predict":         int,
    "gemma_tier":                int,     # 0-6
    # Quality proxies
    "narration_word_count":      int,
    "top_network":               str,     # 'Default'
    "peak_z":                    float,
    "activated_pct":             float,   # % vertices above 1σ
    # Total pipeline
    "total_secs":                float,
}
```

This data feeds:
1. `/api/health` endpoint (already exists in server.py)
2. A Discord `/jemma-benchmark` command showing formatted table
3. Hackathon submission materials (proof of local vs cloud performance)

---

## 6. Phase 2: Internet-Exposed (Month 1–2)

### 6.1 Public Job Submission

```python
# New endpoint in server.py:
@app.post('/api/submit-public')
async def submit_public(
    file: UploadFile,
    email: str = Form(default=''),
    audience: str = Form(default='public'),
):
    """Accept video/audio for public analysis queue."""
    # Rate limit: 1 per IP per hour (no account needed)
    # Priority: 4 (lowest) — Discord Staff still jumps queue
    job_id = f"pub_{int(time.time())}_{uuid4().hex[:8]}"
    # Save to GCS → Cloud Tasks → PC polls
    ...
```

### 6.2 Email Notification

```python
# When job completes, email the submitter:
# "Your brain analysis is ready! View it at: brain.redteamkitchen.com/?r={job_id}"
# Use: sendgrid (free 100/day) or Google Workspace email
```

### 6.3 API Documentation (to gather and link)

| API | URL | Key docs to bookmark |
|---|---|---|
| **Three.js** | threejs.org/docs | BufferGeometry, ShaderMaterial, WebGPURenderer |
| **Ollama REST** | github.com/ollama/ollama/blob/main/docs/api.md | /api/generate, /api/show, keep_alive |
| **Google Cloud Run** | cloud.google.com/run/docs | Concurrency, min-instances, GPU support |
| **Google Cloud Tasks** | cloud.google.com/tasks/docs | Creating tasks, rate limits, retries |
| **Pub/Sub** | cloud.google.com/pubsub/docs | For real-time job notifications |
| **Vertex AI** | cloud.google.com/vertex-ai/docs | For when you graduate from Ollama |
| **Unsloth** | docs.unsloth.ai | Fine-tuning Gemma 4, LoRA, quantization |
| **TRIBE v2** | github.com/gallantlab/TRIBE | CC-BY-NC 4.0 license constraints |
| **Gemma 4** | ai.google.dev/gemma | Multimodal API, vision inputs |
| **GSAP** | gsap.com/docs | Animation in Three.js viewer |
| **Discord.py** | discordpy.readthedocs.io | Slash commands, thread creation |

---

## 7. Phase 3: GCP Hybrid Architecture (Month 2–4)

### Architecture Diagram

```
                    ┌──────────────────────────────────────┐
                    │          brain.redteamkitchen.com    │
                    │  (Cloudflare Tunnel → Squarespace DNS)│
                    └──────────┬───────────────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │          API Gateway               │
              │  (Cloud Run — always on, ~$5/mo)   │
              │  POST /submit → Cloud Tasks queue  │
              └────────────┬──────────────────────┘
                           │
         ┌─────────────────▼─────────────────────┐
         │         Cloud Tasks Queue              │
         │  (Persistent — survives PC outages)    │
         └──────┬──────────────────┬─────────────┘
                │                  │
    ┌───────────▼──────┐    ┌──────▼──────────────────┐
    │  RTX 5090 (home) │    │  GCP L4 GPU VM          │
    │  Primary worker  │    │  (Preemptible fallback)  │
    │  4-7 min/job     │    │  ~$0.80/hr, on-demand   │
    │  Polls queue     │    │  ~10 min/job             │
    └──────────────────┘    └─────────────────────────┘
                │                         │
                └──────────┬──────────────┘
                           │
              ┌────────────▼──────────────┐
              │         GCS Bucket        │
              │  results/, preds/, meta/ │
              └───────────────────────────┘
```

### Cost Model

| Scenario | Monthly Cost |
|---|---|
| Home PC only (24/7) | $0 compute + electricity (~$30) |
| Cloud Tasks queue only | ~$1/month (1M tasks/month free) |
| GCP fallback (100 jobs/month, L4 preemptible) | ~$8/month |
| Cloud Run API (always-on, 1 instance) | ~$5/month |
| GCS storage (10 GB results) | ~$0.23/month |
| **Total hybrid estimate** | **~$14/month** |

### Batch Inference Cost Optimization

```python
# Batch jobs together during off-peak GCP pricing windows
# (Spot instance discounts: 60-90% off on-demand for preemptible)

# GCP Batch API: group 10 jobs → launch 1 VM → process all 10 → shut down
# Cost: 10 × $0.08 (preemptible L4 per job) = $0.80 vs 10 × $0.80 on-demand

BATCH_WINDOW_HOURS = [0, 1, 2, 3, 4, 5]  # midnight–5am cheapest
BATCH_SIZE         = 10                   # trigger batch when 10 jobs queued
```

---

## 8. Gemma 4 Good Hackathon Submission (Deadline: May 18 2026)

**Prize pool:** $200,000 · **Track:** Health & Sciences  
**URL:** kaggle.com/competitions/gemma-4-good

### Why We Win

1. **Novel scientific application** — first public tool using TRIBE v2 + Gemma 4 multimodal
2. **Three-tier audience design** — toddler → researcher, Gemma 4's vision strength
3. **Live demo on Discord** — verifiable, interactive, not just a notebook
4. **Real inference on real RTX 5090 hardware** — not a toy dataset
5. **Alignment with Google's Health & Sciences priority** — cognitive/consumer neuroscience

### Required Deliverables

| Item | Status | Notes |
|---|---|---|
| Kaggle notebook | 📋 Planned | Uses Gemma 4 API (Kaggle provides $300 credit) |
| GitHub repo | 🔧 In progress | TRIBEV2 repo, CC-BY-NC compliant |
| Project description | 📋 Planned | Health & Sciences, ≤500 words |
| Video demo | 📋 Planned | Screen capture of Discord bot + Three.js viewer |
| Gemma 4 usage | ✅ Confirmed | E4B gate + 26B analysis + 31B expert = all model sizes |

### Submission Timeline

- **Week 1 (Apr 21–27):** Cloudflare Tunnel live → public can submit
- **Week 2 (Apr 28–May 4):** Squarespace landing page live, Kaggle notebook draft
- **Week 3 (May 5–11):** Performance benchmark dashboard, video demo recorded
- **Week 4 (May 12–18):** Final submission, repo clean, README complete

---

## 9. Google Account & Developer Programs

### Check Your Accounts

**soumitty@gmail.com:**

1. **Google Cloud credits check:**
   - Go to: `console.cloud.google.com`
   - Sign in with soumitty@gmail.com
   - Go to **Billing → Overview** → look for "Free Trial" or "Credits" 
   - New accounts get **$300 free** for 90 days

2. **Google Developer Program check:**
   - `developers.google.com/community` → check if you have GDE (Google Developer Expert) status
   - `g.dev/soumitty` → see if a profile exists

3. **Google Skills Boost (certifications):**
   - `cloudskillsboost.google.com` → sign in → "My Profile" → see completed courses
   - Check badges at: `google.com/badges`

4. **Google for Startups:**
   - `startup.google.com/intl/us/programs/`
   - **Cloud for Startups**: up to $100K in GCP credits
   - **Google for Startups Accelerator**: $200K GCP credits (AI-focused startups preferred)

### Free Credits to Apply For

| Program | Amount | Eligibility | URL |
|---|---|---|---|
| GCP Free Trial | $300 (90 days) | New accounts | cloud.google.com/free |
| Google for Startups | $100K GCP | LLC/startup registered | startup.google.com |
| NVIDIA Inception | $150K Nebius credits | Apply via inception.nvidia.com | Free program |
| AWS Activate | $5K–$100K | Startup | aws.amazon.com/activate |
| Hugging Face | $2K GPU compute | Open-source projects | huggingface.co/enterprise |
| Google Research Credits | Up to $5K | Academic/research | cloud.google.com/edu |
| NSF SBIR Phase I | ~$275K | Small business, research | seedfund.nsf.gov |

### Google Certifications for LinkedIn

**Most relevant (add within 3 months):**

| Certification | Cost | Time | LinkedIn Value |
|---|---|---|---|
| Google Cloud Professional ML Engineer | $200 | ~3 months study | ⭐⭐⭐⭐⭐ |
| Google Cloud Associate Cloud Engineer | $200 | ~6 weeks | ⭐⭐⭐⭐ |
| TensorFlow Developer Certificate | $100 | 5-hour exam | ⭐⭐⭐⭐ |
| Google Project Management Certificate | $49/month | ~6 months | ⭐⭐⭐ |
| Google Data Analytics Certificate | $49/month | ~6 months | ⭐⭐⭐ |

**Free courses to complete first (CloudSkillsBoost):**
1. "Machine Learning on Google Cloud" → earns a badge
2. "Getting Started with Gemma" → directly relevant
3. "Building LLM-Powered Applications" → ties to hackathon
4. "Introduction to Vertex AI" → for when we move to GCP

**Study path for Professional ML Engineer:**
- `cloudskillsboost.google.com/paths/17` (ML Engineer learning path)
- ~200 hours of content, labs included
- Exam: `webassessor.com/googlecloud` (register with same Google account)

### Internal Google Programs to Check

If soumitty@gmail.com was involved in Google programs (hackathons, developer events, academic programs):
1. Search your gmail for emails from `@google.com` domains about programs/credits
2. Check `myaccount.google.com/linkedaccounts` for any Developer accounts
3. Check `play.google.com/console` → if you registered as a developer ($25 one-time)
4. Check `firebase.google.com` → if you have Firebase projects from before

---

## 10. NSF SBIR Application (Alexios Bluff Mara LLC)

### Target Topics
- **AI1:** Artificial intelligence in computational science (brain signal analysis)
- **AI3:** Conversational AI and language models (Gemma 4 narration)
- **BI2:** Biomedical imaging systems (predicted fMRI)

### Phase I: ~$275,000 (6 months)
**Research question:** "Can TRIBE v2 + Gemma 4 generate clinically meaningful, audience-adaptive brain response narratives from consumer video with accuracy sufficient for educational use?"

**Key deliverables for NSF:**
1. Validation study: compare TRIBE v2 predictions to actual fMRI data for 20 video clips
2. User study: 50 participants across 3 audience tiers rate understanding
3. Software: the Discord bot + Three.js viewer + API
4. IP: provisional patent for the audience-adaptive narration pipeline

### Timeline
- **May 2026:** Submit Phase I via seedfund.nsf.gov (next deadline ~August)
- **November 2026:** Phase I decision
- **Phase II:** $750,000 (if Phase I successful)

---

## 11. NVIDIA + Google Chicago Partnership Path

### NVIDIA Inception Program (Apply Now — Free)
- URL: `nvidia.com/startups`
- Benefits: $150K Nebius compute credits, technical support, co-marketing
- Application: 30 minutes, requires company + product description
- **Relevant because:** RTX 5090 (Blackwell architecture), Ollama, optimized inference

### Google Chicago Offices
- **Midwest HQ:** 1000 W. Fulton Market (100 people, Cloud + Sales teams)
- **Lab:** 210 N. Carpenter St (DeepMind / Research)

**Target contacts:**
- Google Developer Relations → Chicago-based advocates
- Google for Startups → Chicago accelerator cohort (runs annually)
- Google Cloud Enterprise → partnership for healthcare/research

**How small LLCs work with Google:**
1. **Google Cloud Partner Program** → reseller/ISV partner (free to join)
2. **ISV Partner** → list JemmaBrain as a Marketplace solution
3. **Research Partnership** → contact university relations via `research.google`

---

## 12. Technology Integration Checklist (for Hackathon Submission)

```
✅ Gemma 4 E4B       → vision gate classification (173ms, think=False)
✅ Gemma 4 26B MoE   → tiers 0-4 narration (132 tok/s)
✅ Gemma 4 31B dense → tiers 5-6 expert narration (51 tok/s)
✅ Ollama            → local model serving with flash attention
✅ TRIBE v2          → fsaverage5 BOLD prediction (CC-BY-NC 4.0)
✅ Three.js          → PBR brain renderer, Yeo-7 networks, WebGL2
✅ Discord           → RBAC, priority queue, analysis threads, results feed
✅ FastAPI           → REST API + WebSocket for real-time BOLD streaming
✅ GCS               → result persistence + CDN for brain meshes
⬜ Vertex AI         → replace Ollama in GCP production
⬜ Cloud Tasks       → persistent job queue when PC is offline
⬜ Unsloth           → fine-tune Gemma 4 on neuroscience text for better narrations
⬜ WebGPU            → Three.js WebGPU renderer (Chrome 113+)
⬜ Cloudflare Tunnel → public internet access to RTX 5090
⬜ Squarespace       → marketing landing page at redteamkitchen.com
```

---

## 13. Measuring "Ollama Local" vs "GCP Cloud" Performance

Create a comparison table for the hackathon submission:

| Metric | RTX 5090 Local | GCP L4 (g2-standard-4) | GCP A100 |
|---|---|---|---|
| **TRIBE v2 inference** | 4-7 min | ~10 min | ~3 min |
| **Gemma 4 E4B tok/s** | 197 | ~45 | ~120 |
| **Gemma 4 26B tok/s** | 132 | ~28 | ~85 |
| **Gemma 4 31B tok/s** | 51 | ~18 | ~45 |
| **Cost/job** | $0 (electricity only) | ~$0.08 (preemptible) | ~$0.35 |
| **Always available** | ❌ (home PC) | ✅ | ✅ |
| **Latency (first token)** | ~200ms | ~800ms | ~400ms |

**Capture locally with:**
```python
# In ollama_client.py, already logging:
log.debug('[ollama] %s → %d tokens in %.1fs (%.1f tok/s)', ...)

# Add to pipeline.py:
from .pipeline import vram_report
# Returns: {'gpu': 'RTX 5090', 'allocated': X, 'reserved': Y, 'total': 32, 'free': Z}
```

---

## 14. Immediate Action Items (This Week)

### 30 Minutes
1. `winget install Cloudflare.cloudflared` → `cloudflared login` → `cloudflared tunnel create jemmabrain`
2. Add CNAME record in Squarespace DNS: `brain` → `<tunnel-id>.cfargotunnel.com`
3. Run `cloudflared service install` → `net start cloudflared`
4. Test: open `https://brain.redteamkitchen.com/?r=legacy_1776526710`

### 2 Hours
5. Sign into `console.cloud.google.com` with soumitty@gmail.com → check credits
6. Sign into `cloudskillsboost.google.com` → start "Getting Started with Gemma" path
7. Fill out NVIDIA Inception application at `nvidia.com/startups`

### This Week
8. Update `brain.redteamkitchen.com` Squarespace page with "Coming soon" + email signup
9. Record 2-minute screen capture of Discord bot + Three.js viewer → YouTube (unlisted)
10. Start Kaggle notebook draft for Gemma 4 Good hackathon

---

*Last updated: April 2026 · Alexios Bluff Mara LLC*
