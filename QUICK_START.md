# MindScope Quick Start (5 minutes)

Get the full demo running locally with dummy TRIBE v2 data.

## Prerequisites

- Docker & Docker Compose
- Node.js ≥ 20, pnpm
- Python 3.11+

## Start

```bash
# 1. Clone and setup
git clone https://github.com/your-org/mindscope
cd mindscope

# 2. Create environment
cp .env.example .env.local

# 3. Start infrastructure (Postgres, Redis, Ollama)
docker compose up -d

# 4. Install dependencies
make install

# 5. Run development servers in separate terminals:

# Terminal 1: Frontend
make dev-web
# → http://localhost:3000

# Terminal 2: API
make dev-api
# → http://localhost:8000/docs

# Terminal 3: Worker
make dev-worker
# → watches Redis for jobs
```

That's it. The demo is live.

## What You're Looking At

1. **Frontend (port 3000):** Landing page + demo interface
   - Click "Try the Demo" to run the cat clip
   - Returns dummy BOLD activity (random predictions for MVP)
   - Shows placeholder brain activity bars
   - "Ask Gemma" is a chat stub (routes to Ollama in week 2)

2. **API (port 8000):** FastAPI with job management
   - `POST /jobs/demo` — submit preset job
   - `GET /jobs/{job_id}` — poll for status
   - Returns dummy BOLD array (20484 vertices × 40 time points)

3. **Worker (background):** Celery consuming from Redis
   - Simulates TRIBE v2 inference
   - Week 3: will integrate real model from notebook

## Test the API

```bash
# In a 4th terminal
python scripts/test_api.py
```

Should show ✓ all checks passing.

## Next Steps

- **Real TRIBE v2:** Extract from `tribe_v2_5090_ISU_demo.ipynb` into `workers/tribe/serve.py`
- **Gemma chat:** Implement `POST /chat` endpoint calling Ollama
- **Real assets:** Create cat_demo_20s.mp4 with audio + narration
- **Cortex viewer:** Add 3D brain visualization (Week 2)

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for detailed setup and [HACKATHON_PLAN.md](HACKATHON_PLAN.md) for the full roadmap.

## Debugging

```bash
# Check services are running
docker compose ps

# View logs
docker compose logs -f

# Check API is responding
curl http://localhost:8000/health

# Check Redis
redis-cli ping

# Reset database
make db-reset
```

## File Structure

```
mindscope/
├── apps/web/              # Next.js frontend (React)
├── apps/api/              # FastAPI backend (Python)
├── workers/tribe/         # TRIBE v2 inference worker (Celery)
├── docker-compose.yml     # Local services (Postgres, Redis, Ollama)
├── package.json           # Monorepo root
└── Makefile              # Development commands
```

**Questions?** See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed docs.
