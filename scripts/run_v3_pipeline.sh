#!/usr/bin/env bash
# End-to-end v3 pipeline: combine corpus -> train LoRA -> convert to GGUF -> write Modelfile -> smoke test.
# Assumes synth gen has finished and VRAM is free (ollama stopped / no llama-server).
# Run from D:/TRIBEV2 via git-bash.
#
# Flow:
#   1. python combine_brain_narrations.py -> D:/research/datasets/brain_narrations_combined_<N>.jsonl
#   2. python finetune_gemma3_brain.py --dataset <combined> --lora-r 64 --lora-alpha 128 --tag v3-r64
#   3. Create text-only adapter + sanitized base config (same workaround as v2).
#   4. python convert_lora_to_gguf.py -> brain-v3-r64-lora-f16.gguf
#   5. Launch llama-server with --lora on the HF-sourced Q4_K_M base; smoke-test 3 prompts.
#
# Logs go to D:/research/logs/v3_pipeline_<ts>.log.

set -euo pipefail

PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
REPO="D:/TRIBEV2"
RESEARCH="D:/research"
TS="$(date +%s)"
LOGDIR="$RESEARCH/logs"
LOG="$LOGDIR/v3_pipeline_${TS}.log"
mkdir -p "$LOGDIR"

echo "[v3] ts=$TS log=$LOG" | tee -a "$LOG"

# --- Step 1: combine ---
echo "[v3] step 1: combine corpus" | tee -a "$LOG"
"$PY" "$REPO/scripts/combine_brain_narrations.py" 2>&1 | tee -a "$LOG"

COMBINED="$(ls -t "$RESEARCH/datasets"/brain_narrations_combined_*.jsonl 2>/dev/null | head -1)"
if [[ -z "$COMBINED" ]]; then
  echo "[v3] ERROR: combined corpus not found" | tee -a "$LOG"
  exit 2
fi
N_ROWS="$(wc -l < "$COMBINED" | tr -d ' ')"
echo "[v3] combined corpus: $COMBINED  ($N_ROWS rows)" | tee -a "$LOG"

# --- Step 2: v3 finetune (r=64, alpha=128, 3 epochs) ---
echo "[v3] step 2: finetune r=64 alpha=128 epochs=3" | tee -a "$LOG"
"$PY" "$REPO/scripts/finetune_gemma3_brain.py" \
    --dataset "$COMBINED" \
    --epochs 3 \
    --lora-r 64 \
    --lora-alpha 128 \
    --tag v3-r64 2>&1 | tee -a "$LOG"

RUN_DIR="$(ls -td "$RESEARCH/weights"/gemma3-27b-brain-v3-r64-* 2>/dev/null | head -1)"
if [[ -z "$RUN_DIR" ]]; then
  echo "[v3] ERROR: v3 run dir not found" | tee -a "$LOG"
  exit 3
fi
echo "[v3] v3 run dir: $RUN_DIR" | tee -a "$LOG"

# --- Step 3: text-only adapter + sanitized base config ---
echo "[v3] step 3: prep conversion inputs" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, shutil, sys
from pathlib import Path
from safetensors.torch import load_file, save_file

run_dir = sorted(Path('D:/research/weights').glob('gemma3-27b-brain-v3-r64-*'), key=lambda p: p.stat().st_mtime, reverse=True)[0]
final = run_dir / 'final'
textonly = run_dir / 'final_textonly'
textonly.mkdir(parents=True, exist_ok=True)

# vision-strip adapter
sd = load_file(str(final / 'adapter_model.safetensors'))
keep = {k: v for k, v in sd.items() if 'vision_tower' not in k and 'multi_modal_projector' not in k}
save_file(keep, str(textonly / 'adapter_model.safetensors'))
for f in ('adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','processor_config.json'):
    if (final / f).exists():
        shutil.copy2(final / f, textonly / f)
print(f'[v3] wrote {textonly}  kept={len(keep)}/{len(sd)}')

# sanitized base config
BASE_SNAP = Path('C:/Users/soumi/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-bnb-4bit/snapshots/c08b2ba63738aa8cfc60f06741d6356ef4e60b3f')
CFG_ONLY  = Path('D:/research/tmp/gemma3-27b-it-config-only')
CFG_ONLY.mkdir(parents=True, exist_ok=True)
cfg = json.loads((BASE_SNAP / 'config.json').read_text(encoding='utf-8'))
text = cfg.get('text_config') or {}
out = dict(text)
out['architectures'] = ['Gemma3ForCausalLM']
out['model_type']    = 'gemma3_text'
for k in ('bos_token_id','eos_token_id','pad_token_id','torch_dtype'):
    if k not in out and k in cfg: out[k] = cfg[k]
(CFG_ONLY / 'config.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
for f in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
    src = BASE_SNAP / f
    if src.exists():
        shutil.copy2(src, CFG_ONLY / f)
print(f'[v3] config-only base at {CFG_ONLY}')
PY

TEXTONLY="$RUN_DIR/final_textonly"
CFG_BASE="D:/research/tmp/gemma3-27b-it-config-only"

# --- Step 4: convert LoRA -> GGUF f16 ---
echo "[v3] step 4: convert_lora_to_gguf f16" | tee -a "$LOG"
CONVERT="C:/Users/soumi/.unsloth/llama.cpp/convert_lora_to_gguf.py"
OUT_F16="$RUN_DIR/brain-v3-r64-lora-f16.gguf"
"$PY" "$CONVERT" "$TEXTONLY" --base "$CFG_BASE" --outtype f16 --outfile "$OUT_F16" 2>&1 | tee -a "$LOG"

# --- Step 5: Modelfile ---
cat > "$RUN_DIR/Modelfile" <<MF
FROM gemma3:27b
ADAPTER ./$(basename "$OUT_F16")
PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
SYSTEM """You are a neuroscience narration assistant. Given a stimulus description, duration, peak-activity time, and top Schaefer-400 cortical regions by mean |z|, explain what the brain is doing in 3-5 sentences. Group regions into Yeo-7 networks (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default). Be factual, compact, and avoid diagnostic claims. End with a reminder that this is a group-averaged TRIBE v2 prediction, not a diagnostic result."""
MF
echo "[v3] wrote Modelfile -> $RUN_DIR/Modelfile" | tee -a "$LOG"

echo "[v3] DONE. Next: launch llama-server --lora $OUT_F16 and run smoke_llamacpp_brain_v2.py" | tee -a "$LOG"
echo "[v3] or run scripts/run_expanded_eval.py --n 30 after updating ADAPTER path" | tee -a "$LOG"
