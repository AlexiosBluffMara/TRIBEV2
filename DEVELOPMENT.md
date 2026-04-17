# Development Guide for MindScope

This document covers local development setup and the monorepo structure.

## Monorepo Structure

```
mindscope/
├── apps/
│   ├── web/              # Next.js frontend (TypeScript + React)
│   └── api/              # FastAPI backend (Python)
├── workers/
│   ├── tribe/            # TRIBE v2 inference worker (Python + Celery)
│   └── gemma/            # Gemma serving coordination (stub for now)
├── shared/               # Shared TypeScript types and utilities
├── infra/
│   └── terraform/        # Infrastructure as Code for AWS deployment
├── docker-compose.yml    # Local development environment
└── README.md             # Project overview
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js ≥ 20, pnpm ≥ 9
- Python 3.11+
- (Optional) RTX 5090 / H100 for local TRIBE v2 inference

### Local Development

```bash
# 1. Clone and checkout the scaffold branch
git checkout mindscope-scaffold

# 2. Start services (Postgres, Redis, Ollama)
docker compose up -d

# 3. Install dependencies
pnpm install

# 4. Start the development servers (in parallel)
pnpm dev

# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
# Ollama: http://localhost:11434
```

### Environment Variables

Copy `.env.example` to `.env.local` and fill in the secrets:

```bash
cp .env.example .env.local
# Edit .env.local with your Cloudflare R2, Clerk, and other credentials
```

## Frontend (apps/web)

Built with Next.js 15 + React 19 + TypeScript.

```bash
cd apps/web
pnpm dev   # Runs on http://localhost:3000
pnpm build
pnpm lint
```

**Key components:**
- `/app/page.tsx` — Landing page
- `/app/demo/page.tsx` — Demo upload and results
- `/app/api/` — Next.js API routes (auth, job intake)
- `/components/` — Reusable React components

## Backend API (apps/api)

FastAPI Python backend for job orchestration and ML inference.

```bash
cd apps/api
pip install -e ".[dev]"
uvicorn mindscope_api.main:app --reload
```

**Key routes (to be implemented):**
- `POST /jobs` — Submit a prediction job
- `GET /jobs/{job_id}` — Fetch job status
- `WS /ws/jobs/{job_id}` — WebSocket stream of predictions
- `POST /chat` — Chat with Gemma about predictions

## Workers (workers/tribe, workers/gemma)

### TRIBE v2 Worker

Celery task for brain-activity prediction. Extracted from the notebook.

```bash
cd workers/tribe

# Install
pip install -e ".[dev]"

# Run worker (reads jobs from Redis queue)
celery -A serve worker --loglevel=info

# Test a task
celery -A serve call tribe.predict --args '["path/to/video.mp4", "path/to/audio.wav", "narration text", "job-123"]'
```

**Key task:**
- `tribe.predict(video_path, audio_path, text, job_id)` — Returns BOLD predictions

### Gemma Worker (Stub)

Currently Ollama handles Gemma serving directly. This worker will coordinate multi-modal passes and RAG if needed.

## Infrastructure (infra/terraform)

Terraform modules for production deployment on AWS (HIPAA-ready for Carle).

```bash
cd infra/terraform

# Dry-run on local vars
terraform init
terraform plan -var="environment=dev"

# For HIPAA deployment
terraform plan -var="environment=prod" -var="enable_hipaa_mode=true"
```

**Modules:**
- `vpc/` — VPC, subnets, NAT gateways
- `database/` — RDS Postgres with encryption + multi-AZ
- `storage/` — S3 with versioning + server-side logging
- `compute/` — ECS Fargate for GPU workers

## Docker Compose Services

The `docker-compose.yml` provides:

- **postgres** — PostgreSQL 16 (port 5432)
- **redis** — Redis (port 6379)
- **ollama** — Ollama LLM serving (port 11434)

To pre-pull the Gemma model:

```bash
docker compose up -d ollama
docker exec mindscope-ollama ollama pull google/gemma-4-E4B-it
```

## Testing

```bash
# Frontend tests
cd apps/web
pnpm test

# API tests
cd apps/api
pytest

# Worker tests
cd workers/tribe
pytest
```

## Deployment

### Preview Deployment (Vercel)
```bash
git push origin mindscope-scaffold
# GitHub Actions + Vercel will auto-deploy preview
```

### Production Deployment
See `HACKATHON_PLAN.md` section 9 for details. TL;DR:
- Web on Vercel
- API + workers on Fly.io + RunPod GPU nodes
- Databases on Neon (Postgres) + Redis
- Storage on Cloudflare R2

## Debugging

**Frontend:**
- Next.js dev server at http://localhost:3000/__next/debug
- Browser console, React DevTools

**API:**
- FastAPI interactive docs: http://localhost:8000/docs
- Logs via `docker compose logs api`

**Workers:**
- Celery logs: `docker compose logs tribe-worker`
- Flower (Celery monitoring): http://localhost:5555 (if enabled)

**Database:**
- `docker compose exec postgres psql -U mindscope`

**Redis:**
- `docker compose exec redis redis-cli`
- `redis-cli MONITOR` to watch key activity

## Common Tasks

### Add a new dependency

**Frontend:**
```bash
cd apps/web
pnpm add package-name
```

**Backend:**
```bash
cd apps/api
pip install package-name
# Edit pyproject.toml to pin it
```

### Create a new API endpoint

1. Create a router in `apps/api/mindscope_api/routers/`
2. Import and include in `main.py`:
   ```python
   from mindscope_api.routers import jobs
   app.include_router(jobs.router)
   ```

### Extract notebook code to worker

The `tribe_v2_5090_ISU_demo.ipynb` contains the inference logic. To integrate:

1. Copy the feature extraction + prediction cell into `workers/tribe/serve.py`
2. Wrap it in a Celery task
3. Test with the Celery CLI
4. Call from `apps/api` via job submission

## Resources

- [TRIBE v2 GitHub](https://github.com/facebookresearch/tribev2)
- [Ollama docs](https://ollama.ai)
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Next.js docs](https://nextjs.org/docs)
- [Terraform AWS provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
