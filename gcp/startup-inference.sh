#!/usr/bin/env bash
# ============================================================
# Startup script for JemmaBrain GPU inference VMs on GCP
# Runs on each boot of the preemptible g2-standard-8 + L4 instance.
#
# Flow:
#   1. Pull latest bot code from GCS (or git)
#   2. Fetch job parameters from GCS metadata object
#   3. Run TRIBE v2 pipeline on the input video
#   4. Upload results to GCS
#   5. Signal completion (write sentinel + self-terminate)
# ============================================================

set -euo pipefail
LOG="/var/log/jemmabrain-startup.log"
exec > >(tee -a "$LOG") 2>&1

BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket" -H "Metadata-Flavor: Google" 2>/dev/null || echo "jemmabrain-data")
JOB_ID=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/job_id" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
VIDEO_GCS=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/video_gcs" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

echo "=== JemmaBrain Inference VM Startup ==="
echo "Bucket  : gs://$BUCKET"
echo "Job ID  : $JOB_ID"
echo "Video   : $VIDEO_GCS"

# ── Install dependencies ──────────────────────────────────────────────────────
if ! python3 -c "import torch" 2>/dev/null; then
  echo "Installing Python deps..."
  pip3 install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip3 install --quiet nilearn numpy scipy pandas fastapi uvicorn requests google-cloud-storage
fi

# ── Fetch code ────────────────────────────────────────────────────────────────
WORKDIR="/opt/jemmabrain"
if [[ ! -d "$WORKDIR" ]]; then
  mkdir -p "$WORKDIR"
  # Download bot + pipeline code from GCS (sync'd there by setup.sh)
  gsutil -m cp -r "gs://$BUCKET/code/" "$WORKDIR/" || {
    echo "Code not in GCS — cloning from git..."
    git clone https://github.com/YOUR_ORG/jemmabrain.git "$WORKDIR" || true
  }
fi
cd "$WORKDIR"

# ── Download model weights ────────────────────────────────────────────────────
MODEL_DIR="$WORKDIR/models"
mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_DIR/tribe_v2.pt" ]]; then
  echo "Downloading TRIBE v2 weights..."
  gsutil cp "gs://$BUCKET/models/tribe_v2.pt" "$MODEL_DIR/" || \
    echo "WARNING: Model not in GCS. Upload with: gsutil cp models/tribe_v2.pt gs://$BUCKET/models/"
fi

# ── Download input video ──────────────────────────────────────────────────────
if [[ -n "$VIDEO_GCS" ]]; then
  VIDEO_LOCAL="/tmp/input_$(basename "$VIDEO_GCS")"
  echo "Downloading video from $VIDEO_GCS..."
  gsutil cp "$VIDEO_GCS" "$VIDEO_LOCAL"
else
  echo "ERROR: No video_gcs metadata attribute set"
  # Write error sentinel
  echo '{"status":"error","message":"No video_gcs metadata"}' | \
    gsutil cp - "gs://$BUCKET/results/${JOB_ID}_status.json"
  exit 1
fi

# ── Run pipeline ──────────────────────────────────────────────────────────────
echo "Running TRIBE v2 pipeline..."
python3 -m bot.pipeline \
  --input "$VIDEO_LOCAL" \
  --job-id "$JOB_ID" \
  --output-dir "/tmp/results/$JOB_ID" \
  --model-path "$MODEL_DIR/tribe_v2.pt" \
  2>&1 | tee "/tmp/pipeline_${JOB_ID}.log"

PIPELINE_EXIT=${PIPESTATUS[0]}

# ── Upload results to GCS ─────────────────────────────────────────────────────
RESULT_DIR="/tmp/results/$JOB_ID"
if [[ -d "$RESULT_DIR" ]]; then
  echo "Uploading results to gs://$BUCKET/results/$JOB_ID/ ..."
  gsutil -m cp -r "$RESULT_DIR/" "gs://$BUCKET/results/$JOB_ID/"

  # Also upload flattened binary files for the viewer API
  if [[ -f "$RESULT_DIR/${JOB_ID}_bold.bin" ]]; then
    gsutil cp "$RESULT_DIR/${JOB_ID}_bold.bin" "gs://$BUCKET/results/"
    gsutil cp "$RESULT_DIR/${JOB_ID}_meta.json" "gs://$BUCKET/results/"
  fi
fi

# ── Write completion sentinel ─────────────────────────────────────────────────
STATUS="success"
[[ $PIPELINE_EXIT -ne 0 ]] && STATUS="error"
printf '{"status":"%s","job_id":"%s","exit_code":%d}' "$STATUS" "$JOB_ID" "$PIPELINE_EXIT" | \
  gsutil cp - "gs://$BUCKET/results/${JOB_ID}_status.json"

echo "Pipeline finished with status: $STATUS"

# ── Upload pipeline log ───────────────────────────────────────────────────────
gsutil cp "/tmp/pipeline_${JOB_ID}.log" "gs://$BUCKET/results/${JOB_ID}_pipeline.log" 2>/dev/null || true

# ── Self-terminate (preemptible VMs cost money, shut down when done) ──────────
echo "Job complete. Shutting down VM..."
PROJECT=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
INSTANCE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE_META=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)
gcloud compute instances delete "$INSTANCE" --zone="$ZONE_META" --project="$PROJECT" --quiet
