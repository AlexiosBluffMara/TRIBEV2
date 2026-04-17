#!/bin/bash
# Validation script for MindScope local setup
# Checks that all components are properly configured

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "MindScope Setup Validation"
echo "=========================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Docker
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker found"
else
    echo -e "${RED}✗${NC} Docker not found. Install from https://www.docker.com/"
    exit 1
fi

# Docker Compose
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose found"
else
    echo -e "${RED}✗${NC} Docker Compose not found"
    exit 1
fi

# Node
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "${GREEN}✓${NC} Node found ($NODE_VERSION)"
else
    echo -e "${RED}✗${NC} Node.js not found. Install from https://nodejs.org/"
    exit 1
fi

# pnpm
if command -v pnpm &> /dev/null; then
    PNPM_VERSION=$(pnpm -v)
    echo -e "${GREEN}✓${NC} pnpm found (v$PNPM_VERSION)"
else
    echo -e "${RED}✗${NC} pnpm not found. Install with: npm install -g pnpm"
    exit 1
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -V)
    echo -e "${GREEN}✓${NC} Python found ($PYTHON_VERSION)"
else
    echo -e "${RED}✗${NC} Python 3 not found"
    exit 1
fi

echo ""
echo "Checking project structure..."

# Check key directories
dirs=(
    "apps/web"
    "apps/api"
    "workers/tribe"
    "infra/terraform"
    "scripts"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/"
    else
        echo -e "${RED}✗${NC} $dir/ missing"
        exit 1
    fi
done

echo ""
echo "Checking key files..."

files=(
    ".env.example"
    "docker-compose.yml"
    "package.json"
    "pnpm-workspace.yaml"
    "turbo.json"
    "apps/web/package.json"
    "apps/api/pyproject.toml"
    "workers/tribe/pyproject.toml"
    "workers/tribe/serve.py"
    "apps/api/mindscope_api/main.py"
    "apps/api/mindscope_api/routers/jobs.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file missing"
        exit 1
    fi
done

echo ""
echo "Checking Docker services..."

# Start services
echo "Starting Docker services (this may take a minute)..."
docker compose up -d > /dev/null 2>&1 || true

# Wait a bit for services to stabilize
sleep 5

# Check each service
services=("mindscope-postgres" "mindscope-redis" "mindscope-ollama")
for service in "${services[@]}"; do
    if docker ps | grep -q "$service"; then
        echo -e "${GREEN}✓${NC} $service running"
    else
        echo -e "${YELLOW}⚠${NC} $service not running (may still be starting)"
    fi
done

echo ""
echo "Checking service connectivity..."

# Redis
if redis-cli -h localhost ping &> /dev/null; then
    echo -e "${GREEN}✓${NC} Redis responding to ping"
else
    echo -e "${YELLOW}⚠${NC} Redis not responding (may still be starting)"
fi

# Postgres
if PGPASSWORD=mindscope psql -h localhost -U mindscope -d mindscope -c "SELECT 1" &> /dev/null; then
    echo -e "${GREEN}✓${NC} PostgreSQL responding"
else
    echo -e "${YELLOW}⚠${NC} PostgreSQL not responding (may still be starting)"
fi

# Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama responding"
else
    echo -e "${YELLOW}⚠${NC} Ollama not responding (may still be starting)"
fi

echo ""
echo "Validation complete!"
echo ""
echo "Next steps:"
echo "  1. Create .env.local: cp .env.example .env.local"
echo "  2. Install dependencies: make install"
echo "  3. Start services: make infra-up"
echo "  4. Run in separate terminals:"
echo "     - make dev-web"
echo "     - make dev-api"
echo "     - make dev-worker"
echo ""
echo "Then visit: http://localhost:3000"
