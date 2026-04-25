"""Merge the v2 brain LoRA adapter into Gemma-3-27B and export to GGUF for Ollama.

Two-stage export (Unsloth save_pretrained_gguf was failing to wire the two
stages together on this install):

  1. FastLanguageModel load base+adapter, save_pretrained_merged → merged HF bf16 dir
  2. llama.cpp convert_hf_to_gguf.py → bf16 .gguf
  3. llama-quantize.exe bf16.gguf → q4_k_m .gguf
  4. Write Ollama Modelfile pointing at the quantized gguf

Peak disk use on D: during export is ~110 GB (merged HF 55 GB + bf16 gguf 55 GB).
The 17 GB q4_k_m gguf is retained; the intermediates are deleted at the end
unless --keep-intermediates is passed.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/export_brain_gguf.py \\
        --adapter D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/final \\
        --quant q4_k_m
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')

import unsloth  # must import first
from unsloth import FastLanguageModel
import torch


BASE_MODEL       = 'C:/Users/soumi/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-bnb-4bit/snapshots/c08b2ba63738aa8cfc60f06741d6356ef4e60b3f'
MAX_SEQ_LENGTH   = 2048
LLAMA_CPP_DIR    = Path('C:/Users/soumi/.unsloth/llama.cpp')
CONVERT_SCRIPT   = LLAMA_CPP_DIR / 'convert_hf_to_gguf.py'
QUANTIZE_BIN     = LLAMA_CPP_DIR / 'build' / 'bin' / 'Release' / 'llama-quantize.exe'

_SYSTEM_PROMPT = (
    'You are a neuroscience narration assistant. Given a stimulus description, duration, '
    'peak-activity time, and top Schaefer-400 cortical regions by mean |z|, explain what '
    'the brain is doing in 3-5 sentences. Group regions into Yeo-7 networks (Vis, SomMot, '
    'DorsAttn, SalVentAttn, Limbic, Cont, Default). Be factual, compact, and avoid '
    'diagnostic claims. End with a reminder that this is a group-averaged TRIBE v2 prediction, '
    'not a diagnostic result.'
)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f'[export] $ {" ".join(cmd)}')
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if r.returncode != 0:
        raise SystemExit(f'command failed rc={r.returncode}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', type=str, required=True)
    ap.add_argument('--quant',   type=str, default='q4_k_m')
    ap.add_argument('--out-dir', type=str, default='')
    ap.add_argument('--keep-intermediates', action='store_true')
    args = ap.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise SystemExit(f'adapter not found: {adapter_path}')
    if not CONVERT_SCRIPT.exists():
        raise SystemExit(f'convert script missing: {CONVERT_SCRIPT}')
    if not QUANTIZE_BIN.exists():
        raise SystemExit(f'quantize binary missing: {QUANTIZE_BIN}')

    out_dir = Path(args.out_dir) if args.out_dir else adapter_path.parent / f'gguf-{args.quant}'
    merged_dir = out_dir / 'merged_bf16'
    bf16_gguf  = out_dir / f'brain_v2_27b-bf16.gguf'
    quant_gguf = out_dir / f'brain_v2_27b-{args.quant}.gguf'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[export] base       = {BASE_MODEL}')
    print(f'[export] adapter    = {adapter_path}')
    print(f'[export] out_dir    = {out_dir}')
    print(f'[export] merged_dir = {merged_dir}')
    print(f'[export] quant      = {args.quant}')

    # Stage 1: load + merge into bf16 HF dir
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(adapter_path),
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f'[export] load took {time.time()-t0:.1f}s; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB')

    t0 = time.time()
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method = 'merged_16bit',
    )
    print(f'[export] merged bf16 saved in {(time.time()-t0)/60:.1f} min -> {merged_dir}')
    del model
    torch.cuda.empty_cache()

    # Stage 2: convert merged HF to bf16 GGUF
    t0 = time.time()
    _run([
        sys.executable, str(CONVERT_SCRIPT),
        str(merged_dir),
        '--outfile', str(bf16_gguf),
        '--outtype', 'bf16',
    ])
    print(f'[export] bf16 GGUF written in {(time.time()-t0)/60:.1f} min -> {bf16_gguf}')

    # Stage 3: quantize
    t0 = time.time()
    _run([str(QUANTIZE_BIN), str(bf16_gguf), str(quant_gguf), args.quant])
    print(f'[export] {args.quant} GGUF written in {(time.time()-t0)/60:.1f} min -> {quant_gguf}')

    # Stage 4: Modelfile
    modelfile_path = out_dir / 'Modelfile'
    modelfile = (
        f'FROM {quant_gguf.name}\n'
        f'PARAMETER temperature 0.4\n'
        f'PARAMETER top_p 0.9\n'
        f'PARAMETER num_ctx {MAX_SEQ_LENGTH}\n'
        f'SYSTEM """{_SYSTEM_PROMPT}"""\n'
    )
    modelfile_path.write_text(modelfile, encoding='utf-8')
    print(f'[export] wrote Modelfile -> {modelfile_path}')

    # Cleanup
    if not args.keep_intermediates:
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
            print(f'[export] removed {merged_dir}')
        if bf16_gguf.exists():
            bf16_gguf.unlink()
            print(f'[export] removed {bf16_gguf}')

    print('[export] DONE.')
    print(f'[export] to register with Ollama:')
    print(f'         cd "{out_dir}"')
    print(f'         ollama create jemma-brain-v2-27b -f Modelfile')


if __name__ == '__main__':
    main()
