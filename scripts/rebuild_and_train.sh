#!/usr/bin/env bash
# End-to-end self-improvement driver:
#   1. Pull missing P0/P1 neuroscience corpora (skip if already on disk)
#   2. Build a fresh v5 curriculum jsonl
#   3. E4B smoke finetune (shape check, ~10 min)
#   4. 31B curriculum finetune (2 epochs, ~90-120 min)
#   5. lm-eval genuine benchmarks
#   6. tier-control eval
#   7. autoresearch_report refresh
#
# Each step is resumable — if it produced artifacts last time, we reuse them.
#
# Usage:
#   bash scripts/rebuild_and_train.sh [--skip-pull] [--skip-31b] [--skip-e4b-smoke]
#
# Intended for unattended use; all logs go to D:/research/logs/rebuild_<ts>/

set -euo pipefail

SKIP_PULL=""
SKIP_31B=""
SKIP_E4B_SMOKE=""
for arg in "$@"; do
  case "$arg" in
    --skip-pull) SKIP_PULL=1 ;;
    --skip-31b) SKIP_31B=1 ;;
    --skip-e4b-smoke) SKIP_E4B_SMOKE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
TS=$(date +%s)
LOG_DIR="D:/research/logs/rebuild_${TS}"
mkdir -p "$LOG_DIR"

echo "[rebuild] === session ${TS} ==="
echo "[rebuild] log dir: $LOG_DIR"
date

# ---- 1. Pull P0/P1 corpora ----------------------------------------------
if [[ -z "$SKIP_PULL" ]]; then
  echo "[rebuild] === [1/7] pull P0/P1 neuroscience corpora ==="
  "$PY" D:/TRIBEV2/scripts/pull_kaggle_neuro.py \
    --priority "P0,P1" \
    --skip kaggle \
    2>&1 | tee "$LOG_DIR/01_pull.log" || \
    echo "[rebuild] pull step had errors; continuing (check $LOG_DIR/01_pull.log)"
else
  echo "[rebuild] === [1/7] skip pull (--skip-pull) ==="
fi

# ---- 2. Build v5.4 curriculum (full sources + dedup) --------------------
echo "[rebuild] === [2/7] build v5.4 curriculum ==="
DATASET="D:/research/datasets/curriculum_v5_4_${TS}.jsonl"
"$PY" D:/TRIBEV2/scripts/build_curriculum_v4.py \
  --out "$DATASET" \
  --max-per-source 2000 \
  --braingpt-cap 1500 \
  --malikeh-cap 2000 \
  --medmcqa-cap 3000 \
  --asset-cap 2000 \
  --asset-variants 3 \
  --pubmed-oa-cap 2000 \
  --weights "A:1.0,B:0.9,C:1.0,D:0.9" \
  2>&1 | tee "$LOG_DIR/02_build.log"

echo "[rebuild] === [2b/7] validate curriculum ==="
"$PY" D:/TRIBEV2/scripts/check_curriculum.py "$DATASET" \
  --samples-per-source 0 \
  2>&1 | tee "$LOG_DIR/02b_check.log"
if ! grep -q '^\[check\] OK' "$LOG_DIR/02b_check.log"; then
  echo "[rebuild] FATAL: curriculum validation failed" >&2
  exit 1
fi
if [[ ! -s "$DATASET" ]]; then
  echo "[rebuild] FATAL: dataset not produced" >&2
  exit 1
fi
ROWS=$(wc -l < "$DATASET")
echo "[rebuild] dataset: $DATASET ($ROWS rows)"

# ---- 3. E4B smoke -------------------------------------------------------
if [[ -z "$SKIP_E4B_SMOKE" ]]; then
  echo "[rebuild] === [3/7] E4B smoke finetune ==="
  "$PY" D:/TRIBEV2/scripts/finetune_gemma4_curriculum.py \
    --dataset "$DATASET" \
    --base-model "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit" \
    --slug gemma4-e4b-v5 \
    --tag smoke \
    --smoke-steps 40 \
    2>&1 | tee "$LOG_DIR/03_e4b_smoke.log"
else
  echo "[rebuild] === [3/7] skip E4B smoke ==="
fi

# ---- 4. 31B full curriculum finetune ------------------------------------
if [[ -z "$SKIP_31B" ]]; then
  echo "[rebuild] === [4/7] 31B curriculum finetune ==="
  "$PY" D:/TRIBEV2/scripts/finetune_gemma4_curriculum.py \
    --dataset "$DATASET" \
    --epochs 2 \
    --lora-r 64 --lora-alpha 128 \
    --lr 2e-4 \
    --tag v51 \
    --slug gemma4-31b-curriculum-v51 \
    2>&1 | tee "$LOG_DIR/04_31b_train.log"

  ADAPTER_DIR=$(ls -td D:/research/weights/gemma4-31b-curriculum-v51-v51-* 2>/dev/null | head -1)
  if [[ -z "$ADAPTER_DIR" ]]; then
    echo "[rebuild] ERROR: no adapter dir found" >&2
    exit 1
  fi
  FINAL="$ADAPTER_DIR/final"
  echo "[rebuild] adapter at: $FINAL"

  # ---- 5. lm-eval benchmarks -------------------------------------------
  echo "[rebuild] === [5/7] lm-eval benchmarks ==="
  BENCH_SLUG="gemma4-curriculum-v51-$(date +%s)"
  "$PY" D:/TRIBEV2/scripts/run_genuine_benchmarks.py \
    --base-model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
    --variants cur \
    --cur-adapter "$FINAL" \
    --slug "$BENCH_SLUG" \
    --limit 200 \
    2>&1 | tee "$LOG_DIR/05_bench.log" || \
    echo "[rebuild] bench step errored; check $LOG_DIR/05_bench.log"

  # ---- 6. tier-control eval --------------------------------------------
  echo "[rebuild] === [6/7] tier-control eval ==="
  TIER_OUT="D:/research/evals/tier_v51_$(date +%s).csv"
  "$PY" D:/TRIBEV2/scripts/eval_tier_control.py \
    --model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
    --peft "$FINAL" \
    --out "$TIER_OUT" \
    --limit 20 \
    --max-new-tokens 256 \
    2>&1 | tee "$LOG_DIR/06_tier.log" || \
    echo "[rebuild] tier step errored; check $LOG_DIR/06_tier.log"
else
  echo "[rebuild] === [4/7] skip 31B train ==="
fi

# ---- 7. autoresearch report --------------------------------------------
echo "[rebuild] === [7/7] refresh autoresearch report ==="
"$PY" D:/TRIBEV2/scripts/autoresearch_report.py \
  --root D:/research/autoresearch \
  2>&1 | tee "$LOG_DIR/07_report.log" || \
  echo "[rebuild] report step errored (harmless if autoresearch hasn't run yet)"

echo ""
echo "[rebuild] DONE"
echo "[rebuild]   dataset:  $DATASET"
echo "[rebuild]   logs:     $LOG_DIR"
date
