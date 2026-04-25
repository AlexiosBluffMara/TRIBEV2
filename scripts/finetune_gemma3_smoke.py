"""Research-only QLoRA smoke finetune of Gemma-3-27B on the RTX 5090.

Strictly separated from the commercial bot path per docs/LOCAL_FINETUNE_PLAN.md:
  - Base:   unsloth/gemma-3-27b-it-bnb-4bit  (cached locally)
  - Data:   yahma/alpaca-cleaned (public, cleaned Alpaca)
  - Output: D:/research/weights/gemma3-27b-alpaca-smoke-<ts>/   (NOT shipped)

Target Blackwell profile: ~24 GB VRAM, sustained 80-99% GPU util for ~45-90 min.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/finetune_gemma3_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Keep HF cache on C: (SSD system drive), matches user convention
os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')
# Force offline — we have the full 16 GB cached; avoid HF Hub download stalls
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import unsloth  # must import first
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL      = 'C:/Users/soumi/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-bnb-4bit/snapshots/c08b2ba63738aa8cfc60f06741d6356ef4e60b3f'
DATASET         = 'yahma/alpaca-cleaned'
MAX_SEQ_LENGTH  = 2048
LORA_R          = 8
LORA_ALPHA      = 16
MAX_STEPS       = 500          # ~45-90 min smoke test
BATCH_SIZE      = 1
GRAD_ACCUM      = 8
LR              = 2e-4
WARMUP          = 20
LOG_EVERY       = 10
SAVE_EVERY      = 100

ts       = int(time.time())
OUT_DIR  = Path(f'D:/research/weights/gemma3-27b-alpaca-smoke-{ts}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'[finetune] out   = {OUT_DIR}')
print(f'[finetune] base  = {BASE_MODEL}')
print(f'[finetune] data  = {DATASET}')
print(f'[finetune] steps = {MAX_STEPS} (batch {BATCH_SIZE} × grad_accum {GRAD_ACCUM})')
print(f'[finetune] torch = {torch.__version__}  cuda_cap = {torch.cuda.get_device_capability(0)}')
print(f'[finetune] bf16  = {is_bfloat16_supported()}')

# ── Load 4-bit model + tokenizer ──────────────────────────────────────────────
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,       # auto-detect: BF16 on Blackwell
    load_in_4bit   = True,
)
print(f'[finetune] model loaded in {time.time()-t0:.1f}s')

# ── Attach LoRA adapters ──────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r               = LORA_R,
    target_modules  = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],
    lora_alpha      = LORA_ALPHA,
    lora_dropout    = 0.0,
    bias            = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state    = 3407,
    use_rslora      = False,
    loftq_config    = None,
)

# ── Prepare dataset ───────────────────────────────────────────────────────────
t0 = time.time()
ds_raw = load_dataset(DATASET, split='train')
print(f'[finetune] dataset loaded: {len(ds_raw)} rows in {time.time()-t0:.1f}s')

_ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS = tokenizer.eos_token


def _format(batch):
    out = []
    for instr, inp, resp in zip(batch['instruction'], batch['input'], batch['output']):
        out.append(_ALPACA_TEMPLATE.format(instr, inp, resp) + EOS)
    return {'text': out}


ds = ds_raw.map(_format, batched=True, remove_columns=ds_raw.column_names)
print(f'[finetune] dataset formatted: {ds.column_names} ({len(ds)} rows)')

# ── SFT training ──────────────────────────────────────────────────────────────
cfg = SFTConfig(
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_steps    = WARMUP,
    max_steps       = MAX_STEPS,
    learning_rate   = LR,
    logging_steps   = LOG_EVERY,
    save_steps      = SAVE_EVERY,
    optim           = 'adamw_8bit',
    weight_decay    = 0.01,
    lr_scheduler_type = 'linear',
    seed            = 3407,
    output_dir      = str(OUT_DIR),
    save_total_limit = 3,
    report_to       = 'none',
    bf16            = is_bfloat16_supported(),
    fp16            = not is_bfloat16_supported(),
    dataset_text_field = 'text',
    max_length      = MAX_SEQ_LENGTH,
    packing         = False,
    dataset_num_proc = 2,
)

trainer = SFTTrainer(
    model           = model,
    processing_class = tokenizer,
    train_dataset   = ds,
    args            = cfg,
)

print(f'[finetune] trainer ready; VRAM used = {torch.cuda.memory_allocated()/1e9:.2f} GB')

# ── Run ───────────────────────────────────────────────────────────────────────
t_start = time.time()
stats = trainer.train()
elapsed = time.time() - t_start

print(f'\n[finetune] DONE in {elapsed/60:.1f} min')
print(f'[finetune] train_loss = {stats.training_loss:.4f}')
print(f'[finetune] peak VRAM  = {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

# ── Save final adapter ────────────────────────────────────────────────────────
trainer.save_model(str(OUT_DIR / 'final'))
tokenizer.save_pretrained(str(OUT_DIR / 'final'))
print(f'[finetune] adapter saved -> {OUT_DIR / "final"}')
