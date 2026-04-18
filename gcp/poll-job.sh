#!/usr/bin/env bash
# Poll GCS for job completion sentinel, then sync results locally.
#
# Usage:
#   bash gcp/poll-job.sh --job-id job_12345 [--local-dir outputs/results]

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="${BUCKET:-${PROJECT_ID}-data}"
JOB_ID=""
LOCAL_DIR="outputs/results"
POLL_INTERVAL=15

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)    JOB_ID="$2";     shift 2 ;;
    --local-dir) LOCAL_DIR="$2";  shift 2 ;;
    *) shift ;;
  esac
done

[[ -z "$JOB_ID" ]] && { echo "Usage: bash gcp/poll-job.sh --job-id <id>"; exit 1; }

SENTINEL="gs://$BUCKET/results/${JOB_ID}_status.json"

echo "Polling for job $JOB_ID ..."
echo "Sentinel: $SENTINEL"
echo ""

while true; do
  if gsutil -q stat "$SENTINEL" 2>/dev/null; then
    STATUS_JSON=$(gsutil cat "$SENTINEL" 2>/dev/null)
    echo "Job finished: $STATUS_JSON"
    echo ""

    # Sync results to local outputs/results/
    mkdir -p "$LOCAL_DIR"
    echo "Syncing results to $LOCAL_DIR/ ..."
    gsutil cp "gs://$BUCKET/results/${JOB_ID}_meta.json" "$LOCAL_DIR/" 2>/dev/null && echo "  Synced meta.json"
    gsutil cp "gs://$BUCKET/results/${JOB_ID}_bold.bin" "$LOCAL_DIR/" 2>/dev/null && echo "  Synced bold.bin"

    echo ""
    echo "Result available locally at: $LOCAL_DIR/${JOB_ID}_meta.json"
    echo "View in browser: http://localhost:5173/?r=$JOB_ID"
    break
  else
    printf "  [%s] Waiting for job... (checking every ${POLL_INTERVAL}s)\r" "$(date +%H:%M:%S)"
    sleep "$POLL_INTERVAL"
  fi
done
