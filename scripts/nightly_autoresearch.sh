#!/usr/bin/env bash
# Overnight autoresearch driver. Runs scripts/autoresearch_loop.py for N hours,
# auto-restarts on crash (up to MAX_RETRIES), writes a nightly summary.
#
# Usage:
#   bash scripts/nightly_autoresearch.sh [hours] [max-iterations]
#   # defaults: 8 hours, 40 iterations
#
# Intended for unattended overnight use. Each iteration is ~10-25 min
# (E4B smoke finetune + light eval). 8 hours = ~24-48 experiments.

set -euo pipefail

HOURS="${1:-8}"
MAX_ITERS="${2:-40}"
ROOT="D:/research/autoresearch"
PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
MAX_RETRIES=3

TS=$(date +%s)
SESSION_DIR="$ROOT/sessions/nightly_${TS}"
mkdir -p "$SESSION_DIR"
SESSION_LOG="$SESSION_DIR/driver.log"

echo "[nightly] === session ${TS} ===" | tee -a "$SESSION_LOG"
echo "[nightly] budget: ${HOURS}h, max iter: ${MAX_ITERS}" | tee -a "$SESSION_LOG"
echo "[nightly] root:   $ROOT" | tee -a "$SESSION_LOG"
echo "[nightly] log:    $SESSION_LOG" | tee -a "$SESSION_LOG"
date >> "$SESSION_LOG"

# Preflight: ensure the gemma-4 e4b base model is fully cached. A cold
# network pull can take 30-60 min on the first training iter; doing it
# once up-front means every iter starts with a warm cache and the 90-min
# per-iter train timeout never gets eaten by download latency.
BASE_MODEL="unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
echo "[nightly] preflight: warming HF cache for $BASE_MODEL" | tee -a "$SESSION_LOG"
"$PY" - <<PYEOF 2>&1 | tee -a "$SESSION_LOG" || \
  echo "[nightly] preflight download had errors (will retry during loop)" | tee -a "$SESSION_LOG"
import os
os.environ.setdefault('HF_HOME', os.environ.get('HF_HOME', 'D:/unsloth/hf_cache'))
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$BASE_MODEL',
                  allow_patterns=['*.json', '*.safetensors', '*.txt',
                                  'tokenizer.*', 'special_tokens_map.json'])
print('[nightly] preflight: base model cached')
PYEOF

# Sanity: make sure we have at least one prebuilt curriculum to fall back on.
# Prefer v5 (neuro-heavy) over v4.
LATEST_CURR=$(ls -t D:/research/datasets/curriculum_v5_*.jsonl D:/research/datasets/curriculum_v4_*.jsonl 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_CURR" ]]; then
  echo "[nightly] WARN: no curriculum_v{4,5} jsonl in D:/research/datasets — loop will build fresh each iter" | tee -a "$SESSION_LOG"
else
  echo "[nightly] baseline curriculum: $LATEST_CURR" | tee -a "$SESSION_LOG"
fi

TRIES=0
t0=$SECONDS
TOTAL_HOURS="$HOURS"

while (( TRIES < MAX_RETRIES )); do
  REMAINING=$(awk "BEGIN { print $TOTAL_HOURS - ($SECONDS - $t0) / 3600 }")
  if awk "BEGIN { exit !($REMAINING <= 0.25) }"; then
    echo "[nightly] remaining budget ${REMAINING}h too small — stopping" | tee -a "$SESSION_LOG"
    break
  fi
  echo "" | tee -a "$SESSION_LOG"
  echo "[nightly] try $((TRIES+1))/${MAX_RETRIES}  remaining=${REMAINING}h" | tee -a "$SESSION_LOG"

  ARGS=(
    "$PY" D:/TRIBEV2/scripts/autoresearch_loop.py
    --root "$ROOT"
    --hours "$REMAINING"
    --max-iterations "$MAX_ITERS"
    --order newest-first
  )
  if [[ -n "$LATEST_CURR" ]]; then
    ARGS+=(--baseline-dataset "$LATEST_CURR")
  fi

  TRY_LOG="$SESSION_DIR/try_$((TRIES+1)).log"
  echo "[nightly] cmd: ${ARGS[@]}" | tee -a "$SESSION_LOG"
  echo "[nightly] try log: $TRY_LOG" | tee -a "$SESSION_LOG"

  set +e
  "${ARGS[@]}" 2>&1 | tee "$TRY_LOG"
  RC=$?
  set -e
  echo "[nightly] try $((TRIES+1)) finished rc=$RC" | tee -a "$SESSION_LOG"
  TRIES=$((TRIES+1))
  if (( RC == 0 )); then
    echo "[nightly] clean exit" | tee -a "$SESSION_LOG"
    break
  fi
  echo "[nightly] non-zero exit; sleeping 90s before retry" | tee -a "$SESSION_LOG"
  sleep 90
done

echo "" | tee -a "$SESSION_LOG"
echo "[nightly] session done at $(date)" | tee -a "$SESSION_LOG"
echo "[nightly] leaderboard: $ROOT/LEADERBOARD.md" | tee -a "$SESSION_LOG"

# Email-ready summary
if [[ -f "$ROOT/LEADERBOARD.md" ]]; then
  echo "" | tee -a "$SESSION_LOG"
  echo "[nightly] === top 10 ===" | tee -a "$SESSION_LOG"
  head -30 "$ROOT/LEADERBOARD.md" | tee -a "$SESSION_LOG"
fi
