#!/usr/bin/env bash
# Build and push Docker image for JemmaBrain server to Artifact Registry.
# Run from project root after `gcloud auth configure-docker <region>-docker.pkg.dev`
#
# Usage:
#   export PROJECT_ID=jemmabrain-prod
#   export REGION=us-central1
#   bash gcp/build-and-push.sh [--tag v1.2.3]

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/jemmabrain"
TAG="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    *) shift ;;
  esac
done

IMAGE="${REGISTRY}/server:${TAG}"

echo "Building JemmaBrain server image..."
echo "  Registry : $REGISTRY"
echo "  Tag      : $TAG"
echo ""

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build Vite frontend first
echo "[1/3] Building Vite frontend..."
(cd webapp && npm ci --prefer-offline && npm run build)

# Build Docker image
echo "[2/3] Building Docker image..."
docker build -f gcp/Dockerfile.server -t "$IMAGE" -t "${REGISTRY}/server:latest" .

# Push
echo "[3/3] Pushing to Artifact Registry..."
docker push "$IMAGE"
docker push "${REGISTRY}/server:latest"

echo ""
echo "Image pushed: $IMAGE"
echo ""
echo "Deploy to Cloud Run:"
echo "  gcloud run deploy jemmabrain-server \\"
echo "    --image=$IMAGE \\"
echo "    --region=$REGION \\"
echo "    --allow-unauthenticated \\"
echo "    --memory=2Gi --cpu=2"
