#!/usr/bin/env bash
# ============================================================
# Launch a one-shot preemptible GPU VM to run TRIBE v2 inference.
# The VM processes the video, uploads results to GCS, and self-terminates.
#
# Usage:
#   bash gcp/run-inference.sh --video /path/to/video.mp4 [--title "My Video"]
#   bash gcp/run-inference.sh --video gs://my-bucket/uploads/video.mp4
#
# Requirements:
#   gcloud auth application-default login
#   PROJECT_ID and BUCKET env vars (or set defaults below)
# ============================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-${REGION}-a}"
BUCKET="${BUCKET:-${PROJECT_ID}-data}"

VIDEO_PATH=""
TITLE=""
JOB_ID="job_$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)  VIDEO_PATH="$2"; shift 2 ;;
    --title)  TITLE="$2";      shift 2 ;;
    --job-id) JOB_ID="$2";     shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: bash gcp/run-inference.sh --video /path/to/video.mp4"
  exit 1
fi

# ── Upload video to GCS if local ─────────────────────────────────────────────
if [[ "$VIDEO_PATH" != gs://* ]]; then
  FILENAME=$(basename "$VIDEO_PATH")
  GCS_VIDEO="gs://$BUCKET/uploads/${JOB_ID}_${FILENAME}"
  echo "Uploading video to $GCS_VIDEO ..."
  gsutil cp "$VIDEO_PATH" "$GCS_VIDEO"
else
  GCS_VIDEO="$VIDEO_PATH"
fi

echo ""
echo "Launching inference VM..."
echo "  Job ID : $JOB_ID"
echo "  Video  : $GCS_VIDEO"
echo "  Bucket : gs://$BUCKET"
echo ""

# ── Launch preemptible GPU VM ─────────────────────────────────────────────────
INSTANCE_NAME="jemmabrain-inf-${JOB_ID}"

gcloud compute instances create "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-cu124 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True \
  --metadata=bucket="$BUCKET" \
  --metadata=job_id="$JOB_ID" \
  --metadata=video_gcs="$GCS_VIDEO" \
  --metadata=title="$TITLE" \
  --metadata-from-file=startup-script=gcp/startup-inference.sh \
  --service-account="${PROJECT_ID}-bot@${PROJECT_ID}.iam.gserviceaccount.com" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --no-restart-on-failure \
  --quiet

echo ""
echo "VM launched: $INSTANCE_NAME"
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tail -f /var/log/jemmabrain-startup.log"
echo ""
echo "Poll for completion:"
echo "  bash gcp/poll-job.sh --job-id $JOB_ID"
echo ""
echo "Results will appear in:"
echo "  gs://$BUCKET/results/${JOB_ID}_meta.json"
echo "  gs://$BUCKET/results/${JOB_ID}_bold.bin"
