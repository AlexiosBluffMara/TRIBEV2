"""Research-only QLoRA finetune of Gemma-4-31B-IT on synthetic brain-narration data.

Sibling of scripts/finetune_gemma3_brain.py but targets the larger Gemma-4-31B-IT.
Same corpus, same prompt format, same optimizer; scaled memory budget for the
31B backbone on a 32 GB RTX 5090.

Strictly research-only — output flows into the research finetune track, never
into the commercial bot.

Usage:
    # default: r=32, alpha=64, 3 epochs, seq=2048
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/finetune_gemma4_brain.py \
        --dataset D:/research/datasets/brain_narrations_combined_2189.jsonl \
        --tag r32-v2parity

    # larger adapter:
    ... --lora-r 64 --lora-alpha 128 --tag r64-v3parity

    # memory-constrained fallback:
    ... --max-seq-length 1536
"""
from __future__ import annotations

import argparse
import json
import os
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
LORA_R         = 32
LORA_ALPHA     = 64
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LR             = 1.5e-4
WARMUP         = 30
LOG_EVERY      = 5
SAVE_EVERY     = 200


_SYSTEM = (
    "You are a neuroscience narration assistant. Given a stimulus description, duration, "
    "peak-activity time, and top Schaefer-400 cortical regions by mean |z|, explain what "
    "the brain is doing in 3-5 sentences. Group regions into Yeo-7 networks (Vis, SomMot, "
    "DorsAttn, SalVentAttn, Limbic, Cont, Default). Be factual, compact, and avoid "
    "diagnostic claims."
)

_TEMPLATE = """<start_of_turn>user
{system}

{prompt}<end_of_turn>
<start_of_turn>model
{completion}<end_of_turn>"""


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_text_jsonl(rows: list[dict], system: str, template: str, eos: str, out_path: Path) -> None:
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            text = template.format(system=system, prompt=r['prompt'], completion=r['completion']) + eos
            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--base-model', type=str, default=DEFAULT_BASE)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--lora-r', type=int, default=LORA_R)
    ap.add_argument('--lora-alpha', type=int, default=LORA_ALPHA)
    ap.add_argument('--max-seq-length', type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--grad-accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--tag', type=str, default='r32')
    ap.add_argument('--slug', type=str, default='gemma4-31b-brain')
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

    print(f'[g4-ft] out     = {out_dir}')
    print(f'[g4-ft] base    = {args.base_model}')
    print(f'[g4-ft] data    = {data_path}')
    print(f'[g4-ft] epochs  = {args.epochs}  (smoke_steps={args.smoke_steps})')
    print(f'[g4-ft] lora_r  = {args.lora_r}, lora_alpha = {args.lora_alpha}')
    print(f'[g4-ft] seq_len = {args.max_seq_length}  batch = {args.batch_size}  grad_accum = {args.grad_accum}')
    print(f'[g4-ft] torch   = {torch.__version__}  cuda_cap = {torch.cuda.get_device_capability(0)}')
    print(f'[g4-ft] bf16    = {is_bfloat16_supported()}')

    rows = _load_jsonl(data_path)
    print(f'[g4-ft] loaded {len(rows)} rows')

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.base_model,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f'[g4-ft] model loaded in {time.time()-t0:.1f}s')

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
    _write_text_jsonl(rows, _SYSTEM, _TEMPLATE, EOS, text_path)
    ds = load_dataset('json', data_files=str(text_path), split='train')
    print(f'[g4-ft] dataset formatted: {ds.column_names} ({len(ds)} rows) at {text_path}')

    total_steps_per_epoch = max(1, len(ds) // (args.batch_size * args.grad_accum))
    total_steps = total_steps_per_epoch * args.epochs
    print(f'[g4-ft] target total_steps ~= {total_steps}')

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

    print(f'[g4-ft] trainer ready; VRAM used = {torch.cuda.memory_allocated()/1e9:.2f} GB')

    t_start = time.time()
    stats = trainer.train()
    elapsed = time.time() - t_start

    print(f'\n[g4-ft] DONE in {elapsed/60:.1f} min')
    print(f'[g4-ft] train_loss = {stats.training_loss:.4f}')
    print(f'[g4-ft] peak VRAM  = {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

    trainer.save_model(str(out_dir / 'final'))
    tokenizer.save_pretrained(str(out_dir / 'final'))
    print(f'[g4-ft] adapter saved -> {out_dir / "final"}')

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
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'[g4-ft] meta saved  -> {out_dir / "run_meta.json"}')


if __name__ == '__main__':
    main()
