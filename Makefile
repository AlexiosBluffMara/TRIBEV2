.PHONY: help install dev down logs test clean

help:
	@echo "MindScope Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install    Install all dependencies"
	@echo "  make env        Create .env.local from example"
	@echo ""
	@echo "Development:"
	@echo "  make dev        Start all services (Docker + Node + Python)"
	@echo "  make dev-web    Start only frontend (http://localhost:3000)"
	@echo "  make dev-api    Start only API (http://localhost:8000)"
	@echo "  make dev-worker Start only Celery worker"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-up   Start Docker services (Postgres, Redis, Ollama)"
	@echo "  make infra-down Stop Docker services"
	@echo ""
	@echo "Testing & Debugging:"
	@echo "  make test       Run API integration tests"
	@echo "  make logs       Tail Docker service logs"
	@echo "  make db-reset   Reset PostgreSQL database"
	@echo "  make redis      Open Redis CLI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      Remove node_modules, __pycache__, etc"
	@echo ""

install:
	pnpm install
	cd apps/api && pip install -e ".[dev]" && cd ../..
	cd workers/tribe && pip install -e ".[dev]" && cd ../..

env:
	@if [ ! -f .env.local ]; then \
		cp .env.example .env.local; \
		echo "Created .env.local — edit with your secrets"; \
	else \
		echo ".env.local already exists"; \
	fi

infra-up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	docker compose ps

infra-down:
	docker compose down

dev-web:
	cd apps/web && pnpm dev

dev-api:
	cd apps/api && uvicorn mindscope_api.main:app --reload

dev-worker:
	cd workers/tribe && celery -A serve worker --loglevel=info

dev: infra-up
	@echo "Starting all development services..."
	@echo "Frontend:  pnpm --filter=@mindscope/web dev"
	@echo "API:       cd apps/api && uvicorn mindscope_api.main:app --reload"
	@echo "Worker:    cd workers/tribe && celery -A serve worker --loglevel=info"
	@echo ""
	@echo "Frontend: http://localhost:3000"
	@echo "API docs: http://localhost:8000/docs"
	@echo ""
	@echo "Run each command in a separate terminal or use a terminal multiplexer (tmux/screen)"

test:
	python scripts/test_api.py

logs:
	docker compose logs -f

db-reset:
	@echo "Resetting PostgreSQL database..."
	docker compose exec postgres psql -U mindscope -c "DROP DATABASE IF EXISTS mindscope; CREATE DATABASE mindscope;"
	@echo "Done"

redis:
	redis-cli -h localhost

down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info .pytest_cache .ruff_cache
	@echo "Cleaned up build artifacts"
