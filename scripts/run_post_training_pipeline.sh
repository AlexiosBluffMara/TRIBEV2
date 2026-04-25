#!/usr/bin/env bash
# Post-training orchestrator: kicked off manually or by a monitor once both
# overnight Gemma-4-31B runs have finished. Does four things in order:
#   1. GGUF-convert each adapter (r32, r64) via export_gemma4_brain_lora.sh
#   2. Run the Gemma-4 custom three-way eval on the same 30 held-out stimuli
#      used by the Gemma-3 three-way snapshot (via --picks-source)
#   3. Run genuine benchmarks (lm-eval-harness) on 3 variants: g4-base, g4-r32, g4-r64
#   4. Also re-run genuine benchmarks on the Gemma-3 variants for apples-to-apples
#
# Run from D:/TRIBEV2 via git-bash. Logs under D:/research/logs/post_training_<ts>.log.

set -euo pipefail

PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
REPO="D:/TRIBEV2"
RESEARCH="D:/research"
TS="$(date +%s)"
LOG="$RESEARCH/logs/post_training_${TS}.log"
mkdir -p "$RESEARCH/logs"

echo "[post] ts=$TS log=$LOG" | tee -a "$LOG"

R32_DIR="$(ls -td "$RESEARCH/weights"/gemma4-31b-brain-r32-* 2>/dev/null | head -1)"
R64_DIR="$(ls -td "$RESEARCH/weights"/gemma4-31b-brain-r64-* 2>/dev/null | head -1)"

for d in "$R32_DIR" "$R64_DIR"; do
  if [[ -z "$d" ]] || [[ ! -d "$d/final" ]]; then
    echo "[post] missing final/ in run dir: $d" | tee -a "$LOG" >&2
    exit 2
  fi
done

echo "[post] r32 dir: $R32_DIR" | tee -a "$LOG"
echo "[post] r64 dir: $R64_DIR" | tee -a "$LOG"

# --- 1. GGUF conversion ---
echo "[post] step 1: convert adapters to GGUF" | tee -a "$LOG"
bash "$REPO/scripts/export_gemma4_brain_lora.sh" "$R32_DIR" r32 2>&1 | tee -a "$LOG"
bash "$REPO/scripts/export_gemma4_brain_lora.sh" "$R64_DIR" r64 2>&1 | tee -a "$LOG"

R32_GGUF="$R32_DIR/brain-gemma4-r32-lora-f16.gguf"
R64_GGUF="$R64_DIR/brain-gemma4-r64-lora-f16.gguf"

# --- 2. Custom three-way eval on same 30 held-out prompts as Gemma-3 ---
echo "[post] step 2: custom three-way eval on same 30 prompts" | tee -a "$LOG"
PICKS_SRC="$(ls -td "$REPO/outputs/paper/eval_stats_three_way"/*/ 2>/dev/null | head -1)picks.json"
"$PY" "$REPO/scripts/run_gemma4_three_way_eval.py" \
    --n 30 \
    --r32-adapter "$R32_GGUF" \
    --r64-adapter "$R64_GGUF" \
    --picks-source "$PICKS_SRC" 2>&1 | tee -a "$LOG"

# --- 3. Genuine benchmarks on Gemma-4 variants ---
echo "[post] step 3: genuine benchmarks on Gemma-4 variants" | tee -a "$LOG"
"$PY" "$REPO/scripts/run_genuine_benchmarks.py" \
    --slug "gemma4-brain-bench-${TS}" \
    --variants base,r32,r64 \
    --base-model "unsloth/gemma-4-31B-it-unsloth-bnb-4bit" \
    --r32-adapter "$R32_DIR/final" \
    --r64-adapter "$R64_DIR/final" \
    --tasks "arc_challenge,gsm8k,truthfulqa_mc1,piqa,openbookqa" \
    --limit 200 \
    --batch-size 4 2>&1 | tee -a "$LOG"

# --- 4. Genuine benchmarks on Gemma-3 variants (apples-to-apples) ---
V2_DIR="$(ls -td "$RESEARCH/weights"/gemma3-27b-brain-v2-r32-* 2>/dev/null | head -1)"
V3_DIR="$(ls -td "$RESEARCH/weights"/gemma3-27b-brain-v3-r64-* 2>/dev/null | head -1)"
if [[ -n "$V2_DIR" ]] && [[ -n "$V3_DIR" ]]; then
  echo "[post] step 4: genuine benchmarks on Gemma-3 variants" | tee -a "$LOG"
  "$PY" "$REPO/scripts/run_genuine_benchmarks.py" \
      --slug "gemma3-brain-bench-${TS}" \
      --variants base,v2,v3 \
      --base-model "unsloth/gemma-3-27b-it-bnb-4bit" \
      --v2-adapter "$V2_DIR/final" \
      --v3-adapter "$V3_DIR/final" \
      --tasks "arc_challenge,gsm8k,truthfulqa_mc1,piqa,openbookqa" \
      --limit 200 \
      --batch-size 4 2>&1 | tee -a "$LOG"
else
  echo "[post] skipping Gemma-3 benchmarks (missing v2/v3 dirs)" | tee -a "$LOG"
fi

echo "[post] DONE." | tee -a "$LOG"
