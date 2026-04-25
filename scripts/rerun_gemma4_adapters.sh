#!/usr/bin/env bash
# Re-run Gemma-4 r32 + r64 genuine benchmarks after patching adapter_config.json
# to add exclude_modules regex that skips vision_tower/audio_tower (where
# Gemma4ClippableLinear modules live — PEFT doesn't recognize that wrapper as
# a LoRA target).
#
# Run after Stage B Gemma-3 benchmarks finish to avoid GPU contention.
set -euo pipefail

PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
REPO="D:/TRIBEV2"
RESEARCH="D:/research"
TS="$(date +%s)"
LOG="$RESEARCH/logs/gemma4_adapter_rerun_${TS}.log"
mkdir -p "$RESEARCH/logs"

R32_DIR="$(ls -td "$RESEARCH/weights"/gemma4-31b-brain-r32-* 2>/dev/null | head -1)"
R64_DIR="$(ls -td "$RESEARCH/weights"/gemma4-31b-brain-r64-* 2>/dev/null | head -1)"

echo "[gemma4-adapt] log=$LOG ts=$TS" | tee -a "$LOG"
echo "[gemma4-adapt] r32=$R32_DIR" | tee -a "$LOG"
echo "[gemma4-adapt] r64=$R64_DIR" | tee -a "$LOG"

"$PY" "$REPO/scripts/run_genuine_benchmarks.py" \
    --slug "gemma4-adapters-${TS}" \
    --variants r32,r64 \
    --base-model "unsloth/gemma-4-31B-it-unsloth-bnb-4bit" \
    --r32-adapter "$R32_DIR/final" \
    --r64-adapter "$R64_DIR/final" \
    --tasks "arc_challenge,gsm8k,truthfulqa_mc1,piqa,openbookqa" \
    --limit 200 \
    --batch-size 4 2>&1 | tee -a "$LOG"

echo "[gemma4-adapt] DONE." | tee -a "$LOG"
