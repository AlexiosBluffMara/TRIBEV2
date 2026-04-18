# Google Cloud Platform — Complete Service Map for Red Team Kitchen × JemmaBrain

**Vision:** Use every applicable Google product to build a production-grade, revenue-generating, research-grade AI platform around TRIBE v2 + Gemma 4.  This document is a living reference: every GCP service is listed, explained, and mapped to a concrete use case in our stack.

---

## How to Read This Doc

Each section: **What it does → How we use it → Priority** (🟢 Immediate / 🟡 Phase 2 / 🔵 Phase 3 / ⚫ Not applicable)

---

## 1. Compute

### 1.1 Cloud Run  🟢
Serverless containers that scale to zero.  Pay only per-request.

**Our use:** Host `server.py` (FastAPI) as a Cloud Run service.  When the RTX 5090 is offline, the API stays live — it queues jobs to Cloud Tasks and responds to health checks.  Also host the Squarespace webhook handler.

**Config:**
```yaml
# cloudbuild.yaml trigger
gcloud run deploy jemmabrain-api \
  --image gcr.io/PROJECT_ID/jemmabrain-api \
  --region us-central1 \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --allow-unauthenticated
```

---

### 1.2 Cloud Functions (2nd gen)  🟡
Event-driven serverless functions.  Sub-100ms cold start.

**Our use:**
- `on_job_complete` trigger — when a new result lands in GCS, auto-post to Discord `#results-feed`
- `on_youtube_upload` — when a monitored YouTube channel posts a new video, auto-queue it for TRIBE analysis
- Webhook receiver for Squarespace form submissions → Cloud Tasks
- Stripe webhook → update user tier in Firestore

---

### 1.3 Compute Engine (GCE)  🟢 (Phase 2)
Full VMs with GPU attachment.

**Our use:** Preemptible `n1-standard-4` + `nvidia-l4` (24 GB VRAM) for overflow inference.  Costs ~$0.80/hr spot.  Auto-stopped when no jobs pending.  The Cloud Tasks worker checks if the VM is UP; if not, it starts it, waits 90s, then dispatches the job.

```bash
# Start overflow VM (triggered by Cloud Tasks worker)
gcloud compute instances start jemma-l4-worker --zone us-central1-a
```

---

### 1.4 GKE (Google Kubernetes Engine)  🔵
Managed Kubernetes for microservices.

**Our use (Phase 3):** When we grow to >10 concurrent jobs, containerize the TRIBE pipeline into a Kubernetes Job with GPU node pool auto-provisioning.  Each job gets its own pod, result written to GCS, pod deleted.

---

### 1.5 Batch  🟡
Managed batch compute — run thousands of jobs without managing VMs.

**Our use:** Night-window batch inference.  At 2 AM, kick off analysis of all queued YouTube clips, educational videos, ISU study stimuli.  Batch automatically provisions preemptible VMs, runs the jobs, shuts down.  Cost: ~$0.15/job vs $0.80 on-demand.

```python
# Nightly batch job trigger (Cloud Scheduler → Cloud Functions → Batch)
gcloud batch jobs submit jemma-nightly \
  --location us-central1 \
  --config batch_config.json
```

---

## 2. Storage

### 2.1 Cloud Storage (GCS)  🟢
Object storage.  $0.02/GB/month.  Global CDN via Cloud CDN.

**Our use:**
- `gs://jemmabrain-results/` — store `{job_id}_bold.bin` + `{job_id}_meta.json` forever
- `gs://jemmabrain-public/brain.glb` — serve the 3D cortex mesh via CDN (cache forever, one-time upload)
- `gs://jemmabrain-uploads/` — temporary media uploads for cloud-side inference
- `gs://jemmabrain-exports/` — user-downloadable PDF/PNG reports
- Lifecycle rules: delete raw uploads after 7 days; keep results indefinitely

**Already partially integrated** (`bot/gcs_store.py`)

---

### 2.2 Firestore  🟡
Serverless NoSQL document DB.  Real-time sync.

**Our use:**
- User profiles: `users/{discord_user_id}` — tier, job count, email, verification status
- Job metadata: `jobs/{job_id}` — status, timestamps, model used, media URL
- Rate limit state (across Cloud Run instances — shared, consistent)
- ISU collaboration registry: faculty accounts, study IDs

**Why not local SQLite?** Firestore works across Cloud Run replicas without coordination.  Real-time listeners let us push job status to the web UI without polling.

---

### 2.3 Cloud SQL  🔵
Managed PostgreSQL / MySQL.

**Our use (Phase 3):** If we need complex relational queries (e.g., "find all jobs where dominant_network=Default and user_tier=Researcher in the last 30 days"), Cloud SQL with PostgreSQL is the right tool.  Phase 1-2 Firestore is sufficient.

---

### 2.4 BigQuery  🟡
Serverless analytics warehouse.  Query terabytes in seconds with SQL.

**Our use:**
- Analytics: stream every job completion event to BigQuery.  Query: which content types activate which brain networks most?
- YouTube integration: analyze thousands of clips, store BOLD summaries, run cross-video network activation queries
- ISU research: export cohort-level data for academic analysis without exposing raw BOLD vectors
- Google Analytics 4 data export → BigQuery for custom funnel analysis

```sql
-- Example research query in BigQuery
SELECT
  content_category,
  AVG(dominant_network_score) AS avg_default_mode,
  COUNT(*) AS n_clips
FROM `jemmabrain.results.job_events`
WHERE dominant_network = 'Default Mode'
GROUP BY content_category
ORDER BY avg_default_mode DESC
```

---

### 2.5 Bigtable  ⚫
HBase-compatible NoSQL for petabyte-scale time-series.  Overkill for our scale.

---

### 2.6 Spanner  ⚫
Globally distributed relational DB.  Only relevant at Google-scale traffic.

---

## 3. AI / Machine Learning

### 3.1 Vertex AI  🟢
Google's unified ML platform.

**Sub-products we'll use:**

#### 3.1.1 Vertex AI Model Garden  🟢
Pre-trained models ready to deploy via API.

**Our use:** When we migrate away from local Ollama for production:
- Deploy **Gemma 4** (27B) as a Vertex AI endpoint → no local GPU needed for narration
- Compare cost: $0.35/1K tokens vs local RTX 5090 (effectively free)
- Use for burst capacity when local GPU is saturated

```python
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemma-3-27b-it")  # Vertex endpoint
response = model.generate_content("Narrate this BOLD pattern...")
```

#### 3.1.2 Vertex AI Pipelines  🟡
Kubeflow-based ML pipeline orchestration.

**Our use:** Automate the full TRIBE → Analysis → Narration → Export pipeline as a reproducible Vertex Pipeline.  Each step is a containerized component.  Useful for ISU research reproducibility and grant reporting.

#### 3.1.3 Vertex AI Workbench  🟡
Managed JupyterLab on Google Cloud.

**Our use:** ISU faculty get a Workbench instance pre-loaded with our Python SDK.  They can run TRIBE analyses, visualize BOLD data, and write research code without any local setup.

#### 3.1.4 Vertex AI Experiments  🟡
Track ML experiments (hyperparams, metrics) across runs.

**Our use:** Compare narration quality across Gemma 4 model sizes and prompts.  Track which model tier produces highest ISU student comprehension scores.

#### 3.1.5 Vertex AI Feature Store  🔵
Centralized feature repository for ML models.

**Our use (Phase 3):** Store precomputed BOLD feature vectors (per video, per ROI, per network) so future models can use them without re-running TRIBE.

#### 3.1.6 AutoML  🟡
Train custom models without writing model code.

**Our use:** Train a custom content classifier using AutoML Video Intelligence.  Input: short clips.  Output: content category (lecture, music video, news, sports, narrative).  Feed this category into our narration prompt for better context.

---

### 3.2 Document AI  🔵
Extract structured data from PDFs, images, forms.

**Our use (Phase 3):** Parse ISU IRB consent forms, research papers, grant applications automatically.  Extract methodology sections for automated literature review.

---

### 3.3 Natural Language API  🟡
Pre-trained NLP: sentiment, entity extraction, classification.

**Our use:** Analyze the narration text Gemma generates — score reading level (Flesch-Kincaid), sentiment, entity mentions (brain regions named).  Auto-flag narrations that are too technical for the "Student" tier.

---

### 3.4 Speech-to-Text (STT)  🟢
Transcribe audio to text.  Supports 125+ languages.

**Our use:** Before running TRIBE on an audio file, transcribe it to get the textual stimulus.  Feed this transcript to Gemma 4 for the "text-only quick narration" (Stage B of the pipeline) — gives Gemma context about what the brain is responding to.

```python
from google.cloud import speech
client = speech.SpeechClient()
audio  = speech.RecognitionAudio(uri=f"gs://jemmabrain-uploads/{job_id}.mp3")
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
    language_code="en-US",
    enable_automatic_punctuation=True,
)
transcript = client.recognize(config=config, audio=audio)
```

---

### 3.5 Text-to-Speech (TTS)  🟡
Convert text narrations to spoken audio.

**Our use:** Audio versions of the three narration tiers.  A student can click "Listen" and hear the Student-tier narration read aloud.  Also useful for accessibility (ADA compliance for ISU).  WaveNet voices sound near-human.

---

### 3.6 Vision AI (Cloud Vision API)  🟡
Detect objects, faces, text, explicit content in images/video.

**Our use:**
- Pre-screen uploaded videos for explicit/graphic content (safe-search detection) before processing
- Extract keyframe descriptions to supplement Gemma's video understanding
- OCR text visible in educational videos (slides, whiteboard) — feed to narration context

```python
from google.cloud import vision
client = vision.ImageAnnotatorClient()
# Safe-search check
response = client.safe_search_detection(image=image)
likely_nsfw = response.safe_search_annotation.adult >= vision.Likelihood.LIKELY
```

---

### 3.7 Video Intelligence API  🟢
Analyze video: shot detection, labels, objects, speech, text.

**Our use:**
- Extract segment labels from uploaded clips (e.g., "music performance", "lecture", "news broadcast")
- Shot-change detection → correlate with TRIBE temporal peaks
- Transcribe speech in video → Context for Gemma narration
- Content moderation (before TRIBE runs — don't process harmful content)

This is the single most powerful Google AI API for our use case.

---

### 3.8 Recommendations AI  🔵
Personalized product/content recommendations.

**Our use (Phase 3):** Recommend which past brain analyses are most similar to the user's current submission.  "Your clip activated the Default Mode Network like these 5 previously analyzed films."

---

### 3.9 Translation API  🟡
Neural machine translation for 100+ languages.

**Our use:** Translate the three narration tiers into Spanish, French, Mandarin, Japanese for international ISU students and global users.  Auto-detect source language of uploaded audio.

---

## 4. Analytics & Monitoring

### 4.1 Google Analytics 4 (GA4)  🟢
Web analytics — user behavior, events, funnels.

**Implementation plan:**

```html
<!-- In index.html (already partially wired) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

**Custom events to track:**
| Event | Parameters | Use |
|---|---|---|
| `file_submitted` | `{file_type, size_mb, user_tier}` | Funnel start |
| `job_queued` | `{queue_position, model_tier}` | Queue metrics |
| `analysis_complete` | `{duration_s, dominant_network}` | Pipeline success rate |
| `narration_tier_viewed` | `{tier_name, audience_mode}` | Content engagement |
| `brain_viewer_interaction` | `{action: rotate/zoom/network_click}` | 3D UX metrics |
| `audience_mode_changed` | `{from_mode, to_mode}` | Which tiers are used |
| `recipe_intro_dismissed` | `{seconds_shown}` | Intro card engagement |
| `result_shared` | `{platform: twitter/discord/link}` | Viral coefficient |
| `upgrade_clicked` | `{from_tier, to_tier}` | Revenue funnel |
| `isu_account_applied` | — | Partnership conversion |

**GA4 → BigQuery export:** Enable raw event export to BigQuery for custom SQL analysis beyond what GA4 UI shows.

```javascript
// In main.js — track brain interaction
function trackBrainInteraction(action, detail = {}) {
  gtag('event', 'brain_viewer_interaction', {
    action,
    dominant_network: window._currentAnalysis?.dominant_network,
    audience_mode:    window._audienceMode,
    ...detail,
  });
}
```

---

### 4.2 Google Tag Manager (GTM)  🟢
Manage analytics tags without code deploys.

**Our use:** Deploy GA4, Google Ads conversion tracking, LinkedIn Insights Tag, and any future pixels from one place.  ISU can add their own UTM tracking tags for their campaigns without asking us.

---

### 4.3 Looker Studio (formerly Data Studio)  🟡
Business intelligence dashboards.

**Our use:**
- Live dashboard for Soumit: jobs/day, revenue, model usage breakdown
- ISU research dashboard: cohort analysis, network activation heatmaps
- Public "transparency report": how many clips analyzed, top networks, top content categories
- Grant reporting: show NSF reviewers actual usage metrics

Connect to: GA4 + BigQuery + Firestore via built-in connectors.

---

### 4.4 Cloud Monitoring  🟢
Metrics, alerting, dashboards for GCP resources.

**Our use:**
- Alert when RTX 5090 goes offline (Cloudflare tunnel heartbeat)
- Alert when queue depth > 15 (time to start overflow L4 VM)
- Alert when GPU VRAM > 90% (preempt new jobs)
- Alert when Cloud Run error rate > 5%
- Custom metric: `jemma/pipeline_duration_seconds` histogram

```python
from google.cloud import monitoring_v3
# Push custom metric from bot.py after each job
client = monitoring_v3.MetricServiceClient()
series = monitoring_v3.TimeSeries()
series.metric.type = 'custom.googleapis.com/jemma/pipeline_duration'
```

---

### 4.5 Cloud Logging  🟢
Centralized structured log management.

**Our use:**
- Stream bot.py logs to Cloud Logging (already using Python `logging` module)
- Search across all logs: "find all jobs where model_tier=EXPERT and duration > 300s"
- Log-based alerts: "alert when 'ValidationError' appears > 10x/minute" (brute force attempt)
- 30-day retention free tier, then $0.50/GB

```python
# Add to bot/logger.py
from google.cloud.logging import Client as GCPLoggingClient
gcp_logging = GCPLoggingClient()
gcp_logging.setup_logging()  # routes Python logging → Cloud Logging
```

---

### 4.6 Error Reporting  🟢
Automatically group and alert on exceptions.

**Our use:** Any unhandled exception in bot.py or server.py goes to Error Reporting.  Groups by stack trace.  Sends email/PagerDuty alert for first occurrence of new error.

---

### 4.7 Cloud Trace  🔵
Distributed tracing for microservices.

**Our use (Phase 3):** Trace a request from Discord message → bot.py → TRIBE → GCS → WebSocket push.  Identify bottlenecks per stage.

---

### 4.8 Cloud Profiler  🔵
CPU and heap profiler for production Python apps.

**Our use (Phase 3):** Profile TRIBE inference to find Python-level bottlenecks.  Continuous profiling with < 1% overhead.

---

## 5. Networking

### 5.1 Cloud CDN  🟢
Edge caching for GCS-served static assets.

**Our use:** Serve `brain.glb` (the cortex mesh) from CDN instead of our home server.  Users in Europe or Asia get the mesh in < 100ms from a nearby PoP.  Cache `brain.glb` for 24h (it never changes).

Cost: $0.08/GB egress from CDN.  brain.glb is ~8 MB → nearly free.

---

### 5.2 Cloud Load Balancing  🔵
Global HTTP(S) load balancer with SSL termination.

**Our use (Phase 3):** When we have multiple Cloud Run regions + the home PC, route traffic to the nearest healthy backend.  SSL cert managed automatically.

---

### 5.3 Cloud DNS  🟡
Managed DNS.  Low latency, 100% uptime SLA.

**Our use:** Move `redteamkitchen.com` DNS to Cloud DNS from Squarespace DNS.  Create:
- `brain.redteamkitchen.com` → Cloud Run (FastAPI)
- `api.redteamkitchen.com` → Cloud Run (API only, Cloudflare-protected)
- `static.redteamkitchen.com` → GCS bucket (brain.glb CDN)

---

### 5.4 Cloud Armor  🟡
DDoS protection and WAF.

**Our use:** Attach to the Cloud Load Balancer.  Pre-configured rules for OWASP Top 10, bot protection, geo-blocking (block TOR exit nodes, known scanner IPs).  Free preview rules + $0.75/million requests.

---

### 5.5 Cloud NAT  🔵
Outbound NAT for private GCE instances (no public IP on the L4 worker VM).

**Our use:** The overflow L4 VM shouldn't have a public IP.  Cloud NAT gives it internet access for Ollama model pulls.

---

## 6. Application Integration

### 6.1 Cloud Tasks  🟢
Managed task queue with retry, deduplication, scheduling.

**Already in architecture (ROADMAP.md).**  Queues jobs from Cloud Run to the home 5090.  If the 5090 is down, tasks are retried with exponential backoff for up to 24 hours.

```python
from google.cloud import tasks_v2
client = tasks_v2.CloudTasksClient()
parent = client.queue_path('PROJECT', 'us-central1', 'jemma-jobs')
client.create_task(parent=parent, task={'http_request': {
    'http_method': tasks_v2.HttpMethod.POST,
    'url': f'http://{CLOUDFLARE_TUNNEL_URL}/api/submit-internal',
    'body': json.dumps({'job_id': job_id, 'gcs_path': gcs_path}).encode(),
}})
```

---

### 6.2 Cloud Pub/Sub  🟡
Real-time messaging bus.  Millions of messages/second.

**Our use:**
- `jemma-job-complete` topic → multiple subscribers:
  - Cloud Function → Discord results feed post
  - Cloud Function → update Firestore job status
  - BigQuery subscriber → analytics pipeline
- `jemma-youtube-new-video` topic → triggered by YouTube Data API webhook → auto-queue for TRIBE

---

### 6.3 Cloud Scheduler  🟢
Cron jobs managed by Google.

**Already integrated (bot/scheduler.py).**  Extend to:
- 2 AM: trigger nightly Batch inference
- 6 AM: generate weekly "brain network trends" digest → post to Discord
- Daily: pull new YouTube uploads from subscribed channels
- Monthly: generate ISU collaboration report → email to faculty

---

### 6.4 Eventarc  🟡
Event routing from GCP services to Cloud Run / Cloud Functions.

**Our use:** When a new file lands in `gs://jemmabrain-uploads/`, Eventarc automatically triggers a Cloud Function to validate and queue it.  Decouples the upload endpoint from the pipeline.

---

### 6.5 Workflows  🔵
Serverless workflow orchestration (step functions equivalent).

**Our use (Phase 3):** Orchestrate the full pipeline as a Workflow:
1. Validate upload
2. Run STT transcription (Speech-to-Text)
3. Run Video Intelligence
4. Dispatch TRIBE to home PC or L4 VM
5. Wait for completion
6. Run Gemma narration
7. Save to GCS
8. Notify via Pub/Sub

---

### 6.6 Apigee  ⚫
Enterprise API management for large organizations.  Overkill until we have 100+ API consumers.

---

## 7. YouTube Integration  🟢

YouTube is a first-class Google product and a major revenue pathway for us.

### 7.1 YouTube Data API v3
**What it does:** Search videos, get metadata, list channels, manage playlists, get captions.

**Our use:**
- Let users submit a YouTube URL instead of uploading a file
- Auto-download via `yt-dlp` (not Google's API, but works alongside it)
- Pull captions/transcripts for context before TRIBE runs
- Monitor ISU's YouTube channel for new lectures → auto-queue
- Search for educational content by keyword → build analysis dataset

```python
from googleapiclient.discovery import build
yt = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Get captions for a video
captions = yt.captions().list(part='snippet', videoId='dQw4w9WgXcQ').execute()

# Search for educational neuroscience videos
results = yt.search().list(
    part='snippet', q='neuroscience lecture fMRI',
    type='video', videoDuration='medium', maxResults=50
).execute()
```

**User-facing feature:**
```
Submit a YouTube link instead of uploading:
https://brain.redteamkitchen.com/?yt=https://youtube.com/watch?v=...
```
The backend fetches the video, validates it, runs TRIBE, returns the brain analysis.

### 7.2 YouTube Analytics API  🟡
Aggregate analytics for channels you own.

**Our use:** Track engagement on our own Red Team Kitchen YouTube demos.  Which brain visualization videos get the most watch time?  Which explanation tier (Student/Public/Expert) is most replayed?

### 7.3 YouTube Live Streaming  🔵
Real-time stream analysis.

**Our use (Phase 3):** Analyze a live YouTube stream in real-time.  The TRIBE model runs on a rolling 50-second window.  The 3D brain viewer updates live as the stream plays.  Hugely compelling demo for ISU lectures, conferences.

---

## 8. Music Integration  🟡

### 8.1 YouTube Music / Google Play Music API
Currently no public API for streaming music to analyze.  Workaround: users upload audio files (MP3, FLAC, WAV) — which we already support.

### 8.2 Google's MusicLM (Music Generation)
**MusicLM** (via Vertex AI) generates music from text descriptions.  

**Our use:** Generate a custom soundtrack that "matches" the predicted brain activation pattern.  If the Default Mode Network is most active → generate contemplative, slow ambient music.  If Visual cortex is dominant → generate fast, visually-evocative electronic music.  This is a compelling product differentiator.

```python
# Generate brain-matched music (via Vertex AI endpoint)
# "Music that matches: Default Mode Network dominant, Peak at 18s,
#  contemplative video content, low arousal"
music_prompt = f"Ambient meditation music, slow tempo, 60 BPM, {dominant_network} theme"
```

### 8.3 AudioLM
Google's audio language model.  Can continue/generate audio conditioned on a prompt.

**Our use (Phase 3):** Generate personalized audio narrations of brain analyses (vs. text + TTS).

---

## 9. Maps & Location  🔵

### 9.1 Google Maps Platform
Not directly applicable — we're not location-aware.  Could use for visualizing where our users are geographically (combined with GA4 location data).

---

## 10. Developer Tools

### 10.1 Artifact Registry  🟢
Docker image registry + Python package registry.

**Our use:** Store our Docker images (Cloud Run deployments) and potentially our `jemmabrain` Python SDK as a private PyPI package for ISU researchers.

```bash
gcloud artifacts repositories create jemmabrain-docker \
  --repository-format=docker \
  --location=us-central1
```

---

### 10.2 Cloud Build  🟢
CI/CD pipeline.

**Our use:** On `git push` to `main`:
1. Run `pytest` (unit tests)
2. Build Docker image
3. Push to Artifact Registry
4. Deploy to Cloud Run (staging)
5. Run smoke tests
6. Promote to production

Free tier: 120 build-minutes/day.

---

### 10.3 Secret Manager  🟢
Store secrets securely.  Audit trail.

**Our use:** Store `DISCORD_TOKEN`, `HF_TOKEN`, `STRIPE_KEY`, `YOUTUBE_API_KEY`.  Reference in Cloud Run env vars:

```bash
gcloud secrets create discord-token --data-file=.env
gcloud run services update jemmabrain-api \
  --set-secrets=DISCORD_TOKEN=discord-token:latest
```

---

### 10.4 Identity Platform / Firebase Auth  🟡
User authentication.  Sign in with Google, GitHub, email.

**Our use:** Web portal login for `brain.redteamkitchen.com`.  Users log in with Google to get their Verified tier automatically (email verification built in).  ISU users sign in with their ISU Google Workspace account — auto-grants Researcher tier.

---

### 10.5 Identity-Aware Proxy (IAP)  🟡
Protect internal GCP resources with Google login.

**Our use:** The L4 GPU VM's internal API shouldn't be public.  IAP ensures only our Cloud Run service (authenticated service account) can reach it.

---

## 11. Communication

### 11.1 Firebase Cloud Messaging (FCM)  🟡
Push notifications to mobile apps / browsers.

**Our use:** Web push notification when a job completes:
"🧠 Your JemmaBrain analysis is ready! Default Mode Network dominant."

The user can queue a job and close the browser tab.  FCM sends the notification when results are ready.

---

### 11.2 Google Chat API  🔵
Post messages to Google Chat spaces.

**Our use:** Post job completion notifications to ISU's Google Chat workspace (alternative to Discord for faculty who prefer Google ecosystem).

---

### 11.3 Gmail API  🟡
Send transactional email.

**Our use:** Sendgrid alternative.  Send:
- Job completion email with brain PNG attachment
- Researcher tier welcome email
- ISU faculty onboarding email with Discord invite

Using Gmail API through a Google Workspace account (soumitlahiri@philanthropytraders.com) is free up to 500 emails/day.

---

## 12. Security & Compliance

### 12.1 Cloud Key Management Service (KMS)  🟡
Manage encryption keys.

**Our use:** Encrypt BOLD data at rest in GCS with customer-managed keys.  Required for any HIPAA-adjacent research use (if ISU gets IRB approval to run TRIBE on clinical stimuli).

---

### 12.2 Security Command Center  🔵
Vulnerability scanning, threat detection across GCP.

**Our use (Phase 3):** Continuous vulnerability scanning of Cloud Run containers.  Alerts for misconfigured storage buckets, IAM over-permissioning.

---

### 12.3 Cloud DLP (Data Loss Prevention)  🟡
Detect and redact sensitive data in text/images.

**Our use:** Scan narration text before posting publicly — ensure Gemma didn't accidentally include PII from the transcript.  Scan uploaded files for embedded credit card numbers, SSNs.

---

## 13. Productivity (Google Workspace)

### 13.1 Google Docs API  🟡
Create/read/update Docs programmatically.

**Our use:**
- Auto-generate ISU research reports from analysis results
- Push brain analysis summaries to shared Google Docs for faculty review
- Auto-populate grant templates with our metrics

---

### 13.2 Google Slides API  🟡
Create presentations programmatically.

**Our use:** Generate a "Brain Analysis Deck" for each analyzed clip — slide 1: video thumbnail + BOLD heatmap screenshot, slide 2: network breakdown chart, slide 3-5: three narration tiers.  Downloadable by users.

---

### 13.3 Google Sheets API  🟢
Read/write spreadsheets.

**Our use:**
- ISU faculty submit study stimuli via a Google Form → Google Sheets → auto-triggers TRIBE analysis via Cloud Functions
- Track all active ISU researcher accounts in a shared Sheet
- Export BigQuery results to Sheets for faculty who prefer Excel-style analysis

---

### 13.4 Google Forms  🟢
Collect structured user feedback.

**Our use:**
- ISU monthly feedback form (embedded in Discord / emailed to faculty)
- Post-analysis survey: "Did this explanation match your level?" (feeds narration quality training data)
- Grant application interest form for new institutional partners

---

### 13.5 Google Calendar API  🟡
Schedule events programmatically.

**Our use:**
- Auto-create calendar invite when ISU "lunch & learn" is scheduled
- Block off GPU time for ISU study sessions (priority queue + calendar sync)
- Schedule Cloud Scheduler jobs visually from a calendar interface

---

## 14. Firebase (Google-owned)

### 14.1 Firebase Realtime Database  🟡
JSON tree, real-time sync to clients.

**Our use:** Alternative to Firestore WebSocket for simpler real-time job status updates.  `jobs/{job_id}/status` changes → all subscribed browser tabs update instantly.

---

### 14.2 Firebase Hosting  🟡
Static site hosting with global CDN.  Free SSL.

**Our use:** Host the Vite-built Three.js app (`webapp/dist/`) on Firebase Hosting.  Deploy with:
```bash
firebase deploy --only hosting
```
Free on Spark plan.  No servers needed for the frontend.

---

### 14.3 Firebase Remote Config  🔵
Feature flags without app updates.

**Our use:** Toggle audience modes, BOLD colormap options, intro card, announcement banners — without redeploying.  Roll out new features to 10% of users first.

---

### 14.4 Firebase App Check  🔵
Protect APIs from bots and abuse.

**Our use:** Add reCAPTCHA Enterprise to `/api/submit` in the web UI.  Only requests from real browsers pass — blocks automated TRIBE job farm abuse.

---

## 15. Payments

### 15.1 Stripe + Google Pay  🟡
Stripe processes payments.  Google Pay is a payment method.

**Our use:** One-click "Upgrade to Verified" ($5) via Google Pay button.  Conversion rate for Google Pay is typically 20-30% higher than credit card forms.

```javascript
// Google Pay button on upgrade page
const paymentRequest = stripe.paymentRequest({
  country: 'US', currency: 'usd',
  total: { label: 'JemmaBrain Verified Tier', amount: 500 },
  requestPayerEmail: true,
});
```

---

## 16. Google for Startups & Grant Programs

### 16.1 Google for Startups Cloud Program  🟢
**Award:** $100K–$200K in GCP credits (for eligible startups)
**Apply:** cloud.google.com/startup
**Our angle:** AI + neuroscience + academic partnership.  Strong fit.
**Timeline:** 2-4 week application review.

### 16.2 Google Research Credits  🟢
**Award:** $5K–$50K in GCP credits for academic research
**Apply:** research.google/tools/cloud-research-credits/ (submit quarterly)
**Our angle:** ISU co-PI applies.  "TRIBE v2 validation + narration quality study."

### 16.3 NVIDIA Inception (uses GCP)  🟢
NVIDIA Inception members get GCP credits through the NVIDIA + GCP partnership.

### 16.4 Google.org Impact Challenge  🔵
Grants for nonprofits.  If ISU spins this into a 501(c)(3) research initiative — applicable.

---

## 17. Certifications (for Soumit's LinkedIn)

| Certification | Cost | Time | Relevance |
|---|---|---|---|
| Google Cloud Professional ML Engineer | $200 | 4-6 weeks | ⭐⭐⭐⭐⭐ Directly relevant |
| Google Cloud Professional Data Engineer | $200 | 4-6 weeks | ⭐⭐⭐⭐ BigQuery, Pub/Sub |
| Google Cloud Professional Cloud Architect | $200 | 6-8 weeks | ⭐⭐⭐ General GCP |
| TensorFlow Developer Certificate | $100 | 2-4 weeks | ⭐⭐⭐ ML credibility |
| Google Analytics Certification | FREE | 1 week | ⭐⭐⭐⭐ Immediate use |
| Google Cloud Digital Leader | $99 | 1-2 weeks | ⭐⭐ Executive-level overview |

**Recommended order:** GA4 Certification (free, immediate) → ML Engineer → Data Engineer.

**CloudSkillsBoost path for Gemma 4 Good Hackathon:**
1. "Getting Started with Gemma" (free)
2. "Gemma Open Models" learning path
3. "Introduction to Vertex AI" course

---

## 18. Priority Matrix — What to Do Now

| Priority | Service | Action | Cost |
|---|---|---|---|
| 🟢 NOW | GA4 | Add gtag.js to index.html + Squarespace | $0 |
| 🟢 NOW | Cloud Logging | Route Python logs to GCP | $0 (free tier) |
| 🟢 NOW | Cloud Monitoring | Add GPU offline alert | $0 |
| 🟢 NOW | Secret Manager | Move tokens out of .env | $0 |
| 🟢 NOW | YouTube Data API | Accept YouTube URL submissions | $0 |
| 🟢 NOW | Google Sheets API | ISU stimulus submission form | $0 |
| 🟢 NOW | Cloud Storage | Wire gcs_store.py fully | $0.02/GB |
| 🟡 Phase 2 | Speech-to-Text | Transcribe audio before TRIBE | $0.004/15s |
| 🟡 Phase 2 | Video Intelligence | Label content, moderate | $0.10/min |
| 🟡 Phase 2 | Firestore | Replace local job tracking | $0 (free tier) |
| 🟡 Phase 2 | BigQuery | Analytics warehouse | $0 (1 TB free) |
| 🟡 Phase 2 | Firebase Hosting | Deploy Three.js app | $0 |
| 🟡 Phase 2 | Translation API | Multi-language narrations | $0.02/1K chars |
| 🔵 Phase 3 | Vertex AI Gemma | Cloud-based narration | $0.35/1K tokens |
| 🔵 Phase 3 | GKE | Kubernetes job scaling | Variable |

---

*Last updated: 2026-04-18 — Red Team Kitchen × ISU · Recipe No. 001*
