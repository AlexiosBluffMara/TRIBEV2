"""Merge a Gemma-4 LoRA adapter into the base model so lm-eval can load it.

Why: PEFT 0.19.1 rejects `Gemma4ClippableLinear` as a target module because it
is not an `nn.Linear` subclass. Loading via lm-eval's `peft=<adapter>` kwarg
therefore fails during `PeftModel.from_pretrained`. The workaround is to merge
the adapter into the base weights ahead of time and point lm-eval at the merged
directory — no PEFT attach needed at eval time.

Uses Unsloth's FastLanguageModel which handles the ClippableLinear wrapper
correctly (it was the trainer), then `save_pretrained_merged` with
`save_method='merged_4bit'` which writes a ~20 GB HF-loadable bnb-4bit dir.

Usage:
    python scripts/merge_gemma4_adapter_for_eval.py \\
        --adapter D:/research/weights/gemma4-31b-brain-r32-<ts>/final \\
        --out-dir D:/research/weights/gemma4-31b-brain-r32-<ts>/merged_4bit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')

import unsloth  # noqa: F401  must import first
from unsloth import FastLanguageModel
import torch


MAX_SEQ_LENGTH = 2048


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', type=str, required=True,
                    help='path to adapter dir (contains adapter_config.json + safetensors)')
    ap.add_argument('--out-dir', type=str, default='',
                    help='output dir for merged checkpoint (default: <adapter>/../merged_4bit)')
    ap.add_argument('--method', type=str, default='merged_4bit',
                    choices=['merged_4bit', 'merged_16bit'])
    args = ap.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise SystemExit(f'adapter not found: {adapter_path}')

    out_dir = Path(args.out_dir) if args.out_dir else adapter_path.parent / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[merge] adapter = {adapter_path}', flush=True)
    print(f'[merge] out_dir = {out_dir}', flush=True)
    print(f'[merge] method  = {args.method}', flush=True)

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(adapter_path),
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f'[merge] load took {time.time()-t0:.1f}s; '
          f'VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    t0 = time.time()
    model.save_pretrained_merged(
        str(out_dir),
        tokenizer,
        save_method = args.method,
    )
    print(f'[merge] merged save done in {(time.time()-t0)/60:.1f} min -> {out_dir}',
          flush=True)

    del model
    torch.cuda.empty_cache()
    print('[merge] DONE.', flush=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
