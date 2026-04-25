#!/usr/bin/env bash
# One-shot: E4B smoke (shape check) -> 31B full train -> 31B bench -> tier-control.
#
# Usage:
#   bash scripts/run_curriculum_pipeline.sh <dataset-jsonl> [skip_smoke]
#
# Example:
#   bash scripts/run_curriculum_pipeline.sh D:/research/datasets/curriculum_v4_1776684239.jsonl
#   bash scripts/run_curriculum_pipeline.sh D:/research/datasets/curriculum_v4_1776684239.jsonl skip_smoke

set -euo pipefail

DATA="${1:-}"
SKIP_SMOKE="${2:-}"
if [[ -z "$DATA" ]]; then
  echo "usage: $0 <curriculum-jsonl> [skip_smoke]" >&2
  exit 2
fi
if [[ ! -f "$DATA" ]]; then
  echo "dataset not found: $DATA" >&2
  exit 2
fi

PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
LOG_DIR="D:/research/logs/curriculum_pipeline_$(date +%s)"
mkdir -p "$LOG_DIR"
echo "[pipe] log dir: $LOG_DIR"
echo "[pipe] dataset: $DATA"

# ---- 1. E4B smoke (30 steps) --------------------------------------------
if [[ "$SKIP_SMOKE" != "skip_smoke" ]]; then
  echo "[pipe] === E4B smoke (30 steps) ==="
  "$PY" D:/TRIBEV2/scripts/finetune_gemma4_curriculum.py \
    --dataset "$DATA" \
    --base-model "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit" \
    --slug gemma4-e4b-curriculum \
    --tag smoke \
    --smoke-steps 30 \
    2>&1 | tee "$LOG_DIR/01_e4b_smoke.log"
  echo "[pipe] smoke done"
else
  echo "[pipe] === skipping E4B smoke ==="
fi

# ---- 2. 31B full curriculum finetune ------------------------------------
echo "[pipe] === 31B curriculum finetune (2 epochs, r=64, α=128, LR=2e-4) ==="
"$PY" D:/TRIBEV2/scripts/finetune_gemma4_curriculum.py \
  --dataset "$DATA" \
  --epochs 2 \
  --lora-r 64 --lora-alpha 128 \
  --lr 2e-4 \
  --tag cur \
  --slug gemma4-31b-curriculum \
  2>&1 | tee "$LOG_DIR/02_31b_train.log"

# Locate the adapter dir we just wrote
ADAPTER_DIR=$(ls -td D:/research/weights/gemma4-31b-curriculum-cur-* 2>/dev/null | head -1)
if [[ -z "$ADAPTER_DIR" ]]; then
  echo "[pipe] ERROR: no adapter dir found" >&2
  exit 1
fi
FINAL="$ADAPTER_DIR/final"
echo "[pipe] adapter at: $FINAL"

# ---- 3. lm-eval genuine benchmarks --------------------------------------
BENCH_DIR="D:/research/benchmarks/gemma4-curriculum-$(date +%s)"
echo "[pipe] === lm-eval benchmarks -> $BENCH_DIR ==="
BENCH_SLUG=$(basename "$BENCH_DIR")
"$PY" D:/TRIBEV2/scripts/run_genuine_benchmarks.py \
  --base-model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --variants cur \
  --cur-adapter "$FINAL" \
  --slug "$BENCH_SLUG" \
  --limit 200 \
  2>&1 | tee "$LOG_DIR/03_bench.log" || \
  echo "[pipe] bench step returned non-zero (check $LOG_DIR/03_bench.log)"

# ---- 4. tier-control eval -----------------------------------------------
TIER_OUT="D:/research/evals/tier_control_curriculum_$(date +%s).csv"
echo "[pipe] === tier-control eval -> $TIER_OUT ==="
"$PY" D:/TRIBEV2/scripts/eval_tier_control.py \
  --model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --peft "$FINAL" \
  --out "$TIER_OUT" \
  --limit 20 \
  --max-new-tokens 256 \
  2>&1 | tee "$LOG_DIR/04_tier_control.log"

echo ""
echo "[pipe] DONE"
echo "[pipe]   adapter: $FINAL"
echo "[pipe]   bench:   $BENCH_DIR"
echo "[pipe]   tier:    $TIER_OUT"
echo "[pipe]   logs:    $LOG_DIR"
