"""QLoRA finetune of Gemma-4-31B-IT on the multi-signal curriculum corpus.

Sibling of scripts/finetune_gemma4_brain.py, but designed for the v4 curriculum
jsonl (per-row `system` field + signal/tier metadata). Key differences:
  - Reads `system` from each row instead of a single hard-coded prompt
  - Larger default adapter (r=64, alpha=128) because the signal mix is more
    diverse; learning rate tuned to match (2e-4)
  - After save, patches adapter_config.json with `exclude_modules` regex so
    PeftModel.from_pretrained can load the adapter without crashing on
    Gemma4ClippableLinear in the vision/audio towers (see
    memory/feedback_gemma4_peft_clippablelinear.md)

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/finetune_gemma4_curriculum.py \\
        --dataset D:/research/datasets/curriculum_v4_<ts>.jsonl \\
        --tag curriculum-r64

    # E4B smoke run (small base, sanity check shape of the training loss):
    ... --base-model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit --slug gemma4-e4b-curriculum \\
        --smoke-steps 60

All runs are research-only. Output flows into the research weights tree, not
the production bot, until a legal review of the curriculum source licenses is
signed off on (see docs/HACKATHON_STRATEGY.md §4 and docs/DATASET_LEGAL.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')

import unsloth  # must import first
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

import hashlib
import datasets.fingerprint as _fp


def _stable_hash(value) -> str:
    return hashlib.sha256(repr(value).encode('utf-8', errors='replace')).hexdigest()


_fp.Hasher.hash = classmethod(lambda cls, value: _stable_hash(value))
_fp.generate_fingerprint = lambda dataset: _stable_hash(id(dataset))

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


DEFAULT_BASE   = 'unsloth/gemma-4-31B-it-unsloth-bnb-4bit'
MAX_SEQ_LENGTH = 2048
LORA_R         = 64
LORA_ALPHA     = 128
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LR             = 2e-4
WARMUP         = 50
LOG_EVERY      = 5
SAVE_EVERY     = 400

_TEMPLATE = """<start_of_turn>user
{system}

{prompt}<end_of_turn>
<start_of_turn>model
{completion}<end_of_turn>"""

# Applied to adapter_config.json after save so vision/audio towers are skipped
# at load time. Without this, PeftModel.from_pretrained raises
# `Target module Gemma4ClippableLinear(...) is not supported`.
EXCLUDE_MODULES_REGEX = r'.*\.(vision_tower|audio_tower)\..*'


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_text_jsonl(rows: list[dict], template: str, eos: str, out_path: Path) -> None:
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            system = r.get('system') or ''
            text = template.format(system=system, prompt=r['prompt'],
                                   completion=r['completion']) + eos
            meta = {k: r.get(k) for k in ('signal', 'tier', 'source')}
            f.write(json.dumps({'text': text, **meta}, ensure_ascii=False) + '\n')


def _patch_adapter_config(final_dir: Path) -> None:
    cfg_path = final_dir / 'adapter_config.json'
    if not cfg_path.exists():
        print(f'[g4c] WARN: {cfg_path} missing, skipping exclude_modules patch')
        return
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    if cfg.get('exclude_modules'):
        print(f'[g4c] adapter_config already has exclude_modules: {cfg["exclude_modules"]!r}')
        return
    cfg['exclude_modules'] = EXCLUDE_MODULES_REGEX
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(f'[g4c] patched {cfg_path} with exclude_modules={EXCLUDE_MODULES_REGEX!r}')


def _signal_summary(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = f"{r.get('signal', '?')}/{r.get('tier', '?')}"
        out[k] = out.get(k, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--base-model', type=str, default=DEFAULT_BASE)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--lora-r', type=int, default=LORA_R)
    ap.add_argument('--lora-alpha', type=int, default=LORA_ALPHA)
    ap.add_argument('--max-seq-length', type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--grad-accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--tag', type=str, default='curriculum')
    ap.add_argument('--slug', type=str, default='gemma4-31b-curriculum')
    ap.add_argument('--smoke-steps', type=int, default=0,
                    help='if >0, override epochs and stop after N steps (for timing probes)')
    args = ap.parse_args()

    data_path = Path(args.dataset)
    if not data_path.exists():
        raise SystemExit(f'dataset not found: {data_path}')

    ts = int(time.time())
    suffix = f'-{args.tag}' if args.tag else ''
    out_dir = Path(f'D:/research/weights/{args.slug}{suffix}-{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(data_path)
    summary = _signal_summary(rows)

    print(f'[g4c] out     = {out_dir}')
    print(f'[g4c] base    = {args.base_model}')
    print(f'[g4c] data    = {data_path}  ({len(rows)} rows)')
    print(f'[g4c] mix     = {summary}')
    print(f'[g4c] epochs  = {args.epochs}  (smoke_steps={args.smoke_steps})')
    print(f'[g4c] lora_r  = {args.lora_r}, lora_alpha = {args.lora_alpha}')
    print(f'[g4c] seq_len = {args.max_seq_length}  batch = {args.batch_size}  grad_accum = {args.grad_accum}')
    print(f'[g4c] torch   = {torch.__version__}  cuda_cap = {torch.cuda.get_device_capability(0)}')
    print(f'[g4c] bf16    = {is_bfloat16_supported()}')

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.base_model,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f'[g4c] model loaded in {time.time()-t0:.1f}s')

    model = FastLanguageModel.get_peft_model(
        model,
        r               = args.lora_r,
        target_modules  = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        lora_alpha      = args.lora_alpha,
        lora_dropout    = 0.0,
        bias            = 'none',
        use_gradient_checkpointing = 'unsloth',
        random_state    = args.seed,
        use_rslora      = False,
        loftq_config    = None,
    )

    EOS = tokenizer.eos_token

    text_path = out_dir / 'train_text.jsonl'
    _write_text_jsonl(rows, _TEMPLATE, EOS, text_path)
    ds = load_dataset('json', data_files=str(text_path), split='train')
    print(f'[g4c] dataset formatted: {ds.column_names} ({len(ds)} rows) at {text_path}')

    total_steps_per_epoch = max(1, len(ds) // (args.batch_size * args.grad_accum))
    total_steps = total_steps_per_epoch * args.epochs
    print(f'[g4c] target total_steps ~= {total_steps}')

    cfg_kwargs = dict(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        warmup_steps    = WARMUP,
        learning_rate   = args.lr,
        logging_steps   = LOG_EVERY,
        save_steps      = SAVE_EVERY,
        optim           = 'adamw_8bit',
        weight_decay    = 0.01,
        lr_scheduler_type = 'cosine',
        seed            = args.seed,
        output_dir      = str(out_dir),
        save_total_limit = 3,
        report_to       = 'none',
        bf16            = is_bfloat16_supported(),
        fp16            = not is_bfloat16_supported(),
        dataset_text_field = 'text',
        max_length      = args.max_seq_length,
        packing         = False,
        dataset_num_proc = 2,
    )
    if args.smoke_steps > 0:
        cfg_kwargs['max_steps'] = args.smoke_steps
        cfg_kwargs['num_train_epochs'] = 1
    else:
        cfg_kwargs['num_train_epochs'] = args.epochs
    cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model           = model,
        processing_class = tokenizer,
        train_dataset   = ds,
        args            = cfg,
    )

    print(f'[g4c] trainer ready; VRAM used = {torch.cuda.memory_allocated()/1e9:.2f} GB')

    t_start = time.time()
    stats = trainer.train()
    elapsed = time.time() - t_start

    print(f'\n[g4c] DONE in {elapsed/60:.1f} min')
    print(f'[g4c] train_loss = {stats.training_loss:.4f}')
    print(f'[g4c] peak VRAM  = {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

    final_dir = out_dir / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f'[g4c] adapter saved -> {final_dir}')

    _patch_adapter_config(final_dir)

    meta = {
        'base_model': args.base_model,
        'dataset': str(data_path),
        'epochs': args.epochs if args.smoke_steps == 0 else 'smoke',
        'smoke_steps': args.smoke_steps,
        'seed': args.seed,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'max_seq_length': args.max_seq_length,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'lr': args.lr,
        'elapsed_sec': elapsed,
        'train_loss': stats.training_loss,
        'peak_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
        'tag': args.tag,
        'slug': args.slug,
        'ts': ts,
        'mix_counts': summary,
        'exclude_modules': EXCLUDE_MODULES_REGEX,
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'[g4c] meta saved  -> {out_dir / "run_meta.json"}')


if __name__ == '__main__':
    main()
