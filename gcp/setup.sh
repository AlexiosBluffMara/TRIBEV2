#!/usr/bin/env bash
# ============================================================
# JemmaBrain / TRIBE v2  — Google Cloud infrastructure setup
# ============================================================
# Run once after `gcloud auth login && gcloud auth application-default login`
#
# What this creates:
#   - GCS bucket for results, model weights, video uploads
#   - Cloud Run service for the FastAPI viewer server
#   - Cloud Run service for the Discord bot (no GPU needed)
#   - GPU VM template for TRIBE v2 inference jobs
#   - Service accounts & IAM roles
#   - Secret Manager secrets for tokens/keys
#
# Usage:
#   export PROJECT_ID=jemmabrain-prod   # or your existing project
#   export REGION=us-central1
#   bash gcp/setup.sh
# ============================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-jemmabrain-$(date +%s | tail -c5)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-${REGION}-a}"
BUCKET="${BUCKET:-${PROJECT_ID}-data}"

echo "========================================"
echo " JemmaBrain GCP Setup"
echo " Project : $PROJECT_ID"
echo " Region  : $REGION"
echo " Bucket  : gs://$BUCKET"
echo "========================================"
echo ""

# ── 1. Project ───────────────────────────────────────────────────────────────
echo "[1/8] Checking project..."
if ! gcloud projects describe "$PROJECT_ID" &>/dev/null; then
  echo "Creating project $PROJECT_ID..."
  gcloud projects create "$PROJECT_ID" --name="JemmaBrain"
fi
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"

# Enable billing (requires a billing account — set BILLING_ACCOUNT env var)
if [[ -n "${BILLING_ACCOUNT:-}" ]]; then
  gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
fi

# ── 2. Enable APIs ────────────────────────────────────────────────────────────
echo "[2/8] Enabling APIs..."
gcloud services enable --quiet \
  run.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com

# ── 3. GCS Bucket ─────────────────────────────────────────────────────────────
echo "[3/8] Creating GCS bucket gs://$BUCKET ..."
if ! gcloud storage buckets describe "gs://$BUCKET" &>/dev/null; then
  gcloud storage buckets create "gs://$BUCKET" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

# Folder layout inside the bucket
echo "  Bucket layout:"
echo "    gs://$BUCKET/results/      — saved analysis results"
echo "    gs://$BUCKET/uploads/      — raw video uploads"
echo "    gs://$BUCKET/models/       — TRIBE v2 weights (if hosted)"
echo "    gs://$BUCKET/mesh/         — brain.glb + networks.bin (static CDN)"

# Upload static brain mesh files if they exist locally
if [[ -f "webapp/public/brain.glb" ]]; then
  echo "  Uploading brain mesh to GCS..."
  gcloud storage cp webapp/public/brain.glb "gs://$BUCKET/mesh/brain.glb" --cache-control="public, max-age=86400"
  gcloud storage cp webapp/public/networks.bin "gs://$BUCKET/mesh/networks.bin" --cache-control="public, max-age=3600" || true
  # Make mesh files publicly readable
  gcloud storage objects update "gs://$BUCKET/mesh/brain.glb" --add-acl-grant=entity=allUsers,role=READER 2>/dev/null || \
    gcloud storage buckets update "gs://$BUCKET" --no-uniform-bucket-level-access && \
    gsutil acl ch -u AllUsers:R "gs://$BUCKET/mesh/**" 2>/dev/null || true
fi

# ── 4. Service Accounts ───────────────────────────────────────────────────────
echo "[4/8] Creating service accounts..."

# Bot service account (runs Discord bot + inference)
SA_BOT="${PROJECT_ID}-bot@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "${PROJECT_ID}-bot" \
  --display-name="JemmaBrain Bot" 2>/dev/null || true

# Grant roles
for role in roles/storage.objectAdmin roles/secretmanager.secretAccessor \
            roles/logging.logWriter roles/run.invoker; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_BOT" --role="$role" --quiet 2>/dev/null || true
done

# Webapp service account (read-only access to results)
SA_WEB="${PROJECT_ID}-web@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "${PROJECT_ID}-web" \
  --display-name="JemmaBrain Webapp" 2>/dev/null || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_WEB" \
  --role="roles/storage.objectViewer" --quiet 2>/dev/null || true

# ── 5. Secrets ────────────────────────────────────────────────────────────────
echo "[5/8] Creating Secret Manager secrets..."

create_secret() {
  local name=$1
  local value="${2:-REPLACE_ME}"
  if ! gcloud secrets describe "$name" &>/dev/null; then
    printf '%s' "$value" | gcloud secrets create "$name" \
      --replication-policy="automatic" --data-file=-
    echo "  Created secret: $name"
  else
    echo "  Secret already exists: $name (skipped)"
  fi
}

# Load from local .env if present
if [[ -f ".env" ]]; then
  source <(grep -E '^(DISCORD_BOT_TOKEN|DISCORD_GUILD_ID|GEMMA_API_KEY|MOONSHOT_API_KEY|JEMMABRAIN_PUBLIC_URL)=' .env | sed 's/^/export /')
fi

create_secret "discord-bot-token"    "${DISCORD_BOT_TOKEN:-REPLACE_ME}"
create_secret "discord-guild-id"     "${DISCORD_GUILD_ID:-REPLACE_ME}"
create_secret "gemma-api-key"        "${GEMMA_API_KEY:-REPLACE_ME}"
create_secret "moonshot-api-key"     "${MOONSHOT_API_KEY:-REPLACE_ME}"

# ── 6. Artifact Registry (Docker images) ─────────────────────────────────────
echo "[6/8] Creating Artifact Registry repository..."
gcloud artifacts repositories create jemmabrain \
  --repository-format=docker \
  --location="$REGION" \
  --description="JemmaBrain container images" 2>/dev/null || true

export REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/jemmabrain"

# ── 7. Cloud Run — Webapp server ──────────────────────────────────────────────
echo "[7/8] Deploying webapp Cloud Run service..."
# Build and push the server image
gcloud builds submit . \
  --config=gcp/cloudbuild-server.yaml \
  --substitutions="_REGISTRY=$REGISTRY,_BUCKET=$BUCKET" \
  --quiet || echo "  (Build skipped — run manually after Docker image is ready)"

# Deploy Cloud Run (will use image if build succeeded)
IMAGE="${REGISTRY}/server:latest"
if gcloud container images describe "$IMAGE" &>/dev/null 2>&1; then
  gcloud run deploy jemmabrain-server \
    --image="$IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --service-account="$SA_WEB" \
    --port=8765 \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=5 \
    --set-env-vars="GCS_BUCKET=$BUCKET,GCS_MESH_BASE=https://storage.googleapis.com/$BUCKET/mesh" \
    --quiet

  SERVER_URL=$(gcloud run services describe jemmabrain-server --region="$REGION" --format='value(status.url)')
  echo "  Webapp server: $SERVER_URL"
else
  echo "  Webapp image not yet built — run: gcloud builds submit . --config=gcp/cloudbuild-server.yaml"
fi

# ── 8. GPU VM Template for Inference ─────────────────────────────────────────
echo "[8/8] Creating GPU inference VM instance template..."

gcloud compute instance-templates create jemmabrain-gpu \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-cu124 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True \
  --service-account="$SA_BOT" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script=gcp/startup-inference.sh \
  --preemptible \
  --quiet 2>/dev/null || echo "  (GPU template already exists or quota not available)"

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo " Next steps:"
echo "  1. Set billing account:  gcloud billing projects link $PROJECT_ID --billing-account=XXXXX-XXXXX-XXXXX"
echo "  2. Update secrets:       gcloud secrets versions add discord-bot-token --data-file=<(echo YOUR_TOKEN)"
echo "  3. Build Docker images:  bash gcp/build-and-push.sh"
echo "  4. Deploy Cloud Run:     gcloud run deploy jemmabrain-server --image=${REGISTRY}/server:latest ..."
echo "  5. Run inference VM:     bash gcp/run-inference.sh --video /path/to/video.mp4"
echo ""
echo " Project:  $PROJECT_ID"
echo " Bucket:   gs://$BUCKET"
echo " Registry: $REGISTRY"
