#!/usr/bin/env bash
# Convert a Gemma-4-31B QLoRA adapter -> GGUF via convert_lora_to_gguf.py.
# Mirrors run_v3_pipeline.sh step 3-4 but for Gemma-4 (Gemma4ForConditionalGeneration).
# Run from D:/TRIBEV2 via git-bash after the overnight queue has produced a run dir.
#
# Args:
#   $1: adapter run dir (e.g. D:/research/weights/gemma4-31b-brain-r32-<ts>)
#   $2: tag (e.g. r32 or r64) — used in output filename
#
# Output:
#   <run_dir>/final_textonly/                     stripped adapter + tokenizer
#   D:/research/tmp/gemma4-31b-it-config-only/    sanitized text-only base config
#   <run_dir>/brain-gemma4-<tag>-lora-f16.gguf    the GGUF adapter

set -euo pipefail

PY="C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe"
CONVERT="C:/Users/soumi/.unsloth/llama.cpp/convert_lora_to_gguf.py"

RUN_DIR="${1:?usage: $0 <run_dir> <tag>}"
TAG="${2:?usage: $0 <run_dir> <tag>}"

if [[ ! -d "$RUN_DIR/final" ]]; then
  echo "missing: $RUN_DIR/final" >&2
  exit 2
fi

echo "[g4-export] run_dir=$RUN_DIR  tag=$TAG"

# --- Prep: strip vision tensors from adapter + build text-only base config ---
"$PY" - <<PY
import json, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

run_dir = Path(r"$RUN_DIR")
final   = run_dir / 'final'
textonly= run_dir / 'final_textonly'
textonly.mkdir(parents=True, exist_ok=True)

sd = load_file(str(final / 'adapter_model.safetensors'))
keep = {k: v for k, v in sd.items()
        if 'vision_tower' not in k
        and 'multi_modal_projector' not in k
        and 'audio_tower' not in k
        and 'embed_vision' not in k
        and 'embed_audio' not in k}
save_file(keep, str(textonly / 'adapter_model.safetensors'))
for f in ('adapter_config.json','tokenizer.json','tokenizer_config.json',
          'chat_template.jinja','processor_config.json','special_tokens_map.json'):
    src = final / f
    if src.exists():
        shutil.copy2(src, textonly / f)
print(f'[g4-export] adapter: kept {len(keep)}/{len(sd)} tensors -> {textonly}')

BASE_SNAP = Path(r"D:/unsloth/hf_cache/hub/models--unsloth--gemma-4-31B-it-unsloth-bnb-4bit/snapshots/ffd2077bfbce1d04f37918c280354b70a52fee04")
CFG_ONLY  = Path(r"D:/research/tmp/gemma4-31b-it-config-only")
CFG_ONLY.mkdir(parents=True, exist_ok=True)

cfg = json.loads((BASE_SNAP / 'config.json').read_text(encoding='utf-8'))
text = dict(cfg.get('text_config') or {})
text['architectures'] = ['Gemma4ForConditionalGeneration']
for k in ('bos_token_id','eos_token_id','pad_token_id','torch_dtype'):
    if k not in text and k in cfg: text[k] = cfg[k]
# drop vision/audio/multimodal blocks
for k in ('vision_config','audio_config','quantization_config',
          'vision_soft_tokens_per_image','boi_token_id','eoi_token_id',
          'audio_token_id','eoa_token_id','eoa_token_index','image_token_id',
          'video_token_id','boa_token_id'):
    text.pop(k, None)
(CFG_ONLY / 'config.json').write_text(json.dumps(text, indent=2), encoding='utf-8')
for f in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
    src = BASE_SNAP / f
    if src.exists():
        shutil.copy2(src, CFG_ONLY / f)
print(f'[g4-export] text-only base config -> {CFG_ONLY}')
PY

TEXTONLY="$RUN_DIR/final_textonly"
CFG_BASE="D:/research/tmp/gemma4-31b-it-config-only"
OUT_F16="$RUN_DIR/brain-gemma4-${TAG}-lora-f16.gguf"

echo "[g4-export] convert_lora_to_gguf -> $OUT_F16"
"$PY" "$CONVERT" "$TEXTONLY" --base "$CFG_BASE" --outtype f16 --outfile "$OUT_F16"

ls -lh "$OUT_F16"
echo "[g4-export] DONE"
