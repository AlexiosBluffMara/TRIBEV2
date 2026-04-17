# Local Development Setup

This guide walks through setting up MindScope locally with all services running.

## Prerequisites

- **Docker & Docker Compose** — for Postgres, Redis, Ollama
- **Node.js ≥ 20, pnpm ≥ 9** — for frontend
- **Python 3.11+** — for backend and workers
- **Git** — for version control
- **(Optional) GPU** — for local TRIBE v2 inference (RTX 5090 / H100 recommended)

## Step 1: Environment Setup

Clone and set up the repository:

```bash
git clone https://github.com/your-org/mindscope
cd mindscope
git checkout mindscope-scaffold
```

Create environment file:

```bash
cp .env.example .env.local
```

Edit `.env.local` with your local paths (for development, defaults work fine):

```bash
# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WEBSOCKET_URL=ws://localhost:8000

# Backend
DATABASE_URL=postgresql://mindscope:mindscope@localhost:5432/mindscope
REDIS_URL=redis://localhost:6379

# TRIBE v2 (local GPU optional)
TRIBE_DEVICE=cuda:0  # or cpu for CPU-only testing

# Ollama (will run in Docker)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=google/gemma-4-E4B-it
```

## Step 2: Start Infrastructure Services

Start Postgres, Redis, and Ollama:

```bash
docker compose up -d

# Wait for services to be healthy
docker compose ps
```

Verify services are running:

```bash
# Check Redis
redis-cli ping  # Should return "PONG"

# Check Postgres
psql -U mindscope -h localhost -d mindscope -c "SELECT version();"

# Check Ollama (will be empty until first model pull)
curl http://localhost:11434/api/tags
```

**Pre-pull Gemma model** (optional, will auto-download on first use):

```bash
docker exec mindscope-ollama ollama pull google/gemma-4-E4B-it
```

## Step 3: Install Dependencies

### Frontend

```bash
pnpm install
pnpm --filter=@mindscope/web install
```

### Backend

```bash
cd apps/api
pip install -e ".[dev]"
cd ../..
```

### Workers

```bash
cd workers/tribe
pip install -e ".[dev]"
cd ../..
```

## Step 4: Create Demo Assets

### Quick Test (Dummy Assets)

For quick testing without actual video files, the system will use dummy data:

```bash
# Create assets directory
mkdir -p assets
touch assets/cat_demo_20s.mp4
touch assets/cat_demo_20s_audio.wav
```

The Celery worker will return random feature arrays regardless of actual file content (for now).

### Real Demo Assets (Optional)

To use actual video with brain activity predictions:

1. **Source a cat video** (20 seconds recommended)
2. **Extract audio and create narration:**

```bash
# Extract audio from video
ffmpeg -i your_cat_video.mp4 -q:a 9 -n audio.mp3

# Record narration or use TTS
# Example with Google Cloud TTS:
gcloud text-to-speech --text="A tabby cat sits on a windowsill, stretching." \
  --voice-gender=FEMALE --output-file narration.wav

# Mix audio + narration
ffmpeg -i audio.mp3 -i narration.wav -filter_complex "[0]volume=0.7[a];[1]volume=0.5[b];[a][b]amix=inputs=2:duration=first[out]" \
  -map "[out]" -q:a 9 assets/cat_demo_20s_audio.wav
```

3. **Place files in expected locations:**

```bash
mv your_cat_video.mp4 assets/cat_demo_20s.mp4
# audio already at assets/cat_demo_20s_audio.wav
```

## Step 5: Run Development Servers

### Terminal 1: Frontend

```bash
cd apps/web
pnpm dev
# Runs on http://localhost:3000
```

### Terminal 2: API

```bash
cd apps/api
uvicorn mindscope_api.main:app --reload
# API docs at http://localhost:8000/docs
# Health check: curl http://localhost:8000/health
```

### Terminal 3: Celery Worker

```bash
cd workers/tribe
celery -A serve worker --loglevel=info
# Watches Redis for jobs
```

## Step 6: Test the Demo

Once all services are running:

### Via Browser

1. Open http://localhost:3000
2. Click "Try the Demo" on the landing page
3. Watch the progress bar fill
4. See results appear in the right panel

### Via API

```bash
# Run the test suite
python scripts/test_api.py

# Or manually:

# Submit demo job
curl -X POST http://localhost:8000/jobs/demo \
  -H "Content-Type: application/json" \
  -d '{}'

# Get job status (replace JOB_ID)
curl http://localhost:8000/jobs/{JOB_ID}
```

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker ps

# Restart everything
docker compose down
docker compose up -d
```

### Redis Connection Error

```bash
# Check Redis is accessible
redis-cli -h localhost ping

# If docker Redis, check logs
docker compose logs redis
```

### Celery Worker Won't Connect

```bash
# Verify REDIS_URL in .env.local
# Test connection:
redis-cli -h localhost PING

# Check Celery logs:
celery -A workers.tribe.serve worker --loglevel=debug
```

### API Import Errors

```bash
# Reinstall dependencies
cd apps/api
pip install -e ".[dev]" --force-reinstall
```

### Frontend Can't Reach API

```bash
# Verify API is running
curl http://localhost:8000/health

# Check .env.local has correct NEXT_PUBLIC_API_URL
# Frontend must rebuild after .env changes:
cd apps/web && pnpm dev
```

## Next Steps

- **Integrate actual TRIBE v2:** Extract inference code from `tribe_v2_5090_ISU_demo.ipynb` into `workers/tribe/serve.py`
- **Enable Gemma chat:** Implement `/chat` endpoint in FastAPI that calls Ollama
- **Add WebSocket streaming:** Real-time result updates to frontend
- **Deploy preview:** Push to GitHub to trigger Vercel preview deployment

## Useful Commands

```bash
# Monitor Celery tasks
celery -A serve events

# Check Redis keys
redis-cli KEYS '*'

# Tail logs
docker compose logs -f postgres redis ollama

# Reset database
docker compose exec postgres psql -U mindscope -c "DROP DATABASE mindscope; CREATE DATABASE mindscope;"

# View API documentation
open http://localhost:8000/docs
```

## Architecture Reminder

```
Browser (3000)
    ↓ (HTTP/WS)
Next.js App
    ↓ (HTTP)
FastAPI (8000) → Redis → Celery Worker → TRIBE v2 (GPU) → R2 Storage
      ↓
PostgreSQL (5432)
      ↓
Ollama (11434) → Gemma 4 E4B
```

All communication goes through localhost during development. Docker Compose ensures services are networked correctly.
