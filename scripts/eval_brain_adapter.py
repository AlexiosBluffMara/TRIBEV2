"""Research-only eval: compare base Gemma-3-27B vs brain-LoRA adapter outputs.

Loads the base model + adapter produced by scripts/finetune_gemma3_brain.py,
samples a handful of held-out brain-narration prompts, and prints side-by-side
completions so we can eyeball whether the LoRA shifted the style toward our
clinician-paragraph target.

Two-phase design: both the base model and the adapter model are ~24 GB 4-bit
quantized; on a 32 GB 5090 they cannot coexist in VRAM. `del model; empty_cache`
leaves enough bnb state that the second from_pretrained still dispatches to CPU.
So the `both` orchestrator forks each phase into its own Python subprocess —
process death is the only bulletproof way to release bnb quantized VRAM.

Usage:
    # Orchestrated side-by-side:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/eval_brain_adapter.py \\
        --adapter D:/research/weights/gemma3-27b-brain-<ts>/final \\
        --dataset D:/research/datasets/brain_narrations_<ts>.jsonl \\
        --n 3

    # Or a single phase (internal use by the orchestrator):
    ... --phase base --picks-file picks.json --out base.json
    ... --phase adapter --picks-file picks.json --out adapter.json --adapter <path>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


BASE_MODEL     = 'C:/Users/soumi/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-bnb-4bit/snapshots/c08b2ba63738aa8cfc60f06741d6356ef4e60b3f'
MAX_SEQ_LENGTH = 2048

_SYSTEM = (
    "You are a neuroscience narration assistant. Given a stimulus description, duration, "
    "peak-activity time, and top Schaefer-400 cortical regions by mean |z|, explain what "
    "the brain is doing in 3-5 sentences. Group regions into Yeo-7 networks (Vis, SomMot, "
    "DorsAttn, SalVentAttn, Limbic, Cont, Default). Be factual, compact, and avoid "
    "diagnostic claims."
)

_CHAT_TEMPLATE = """<start_of_turn>user
{system}

{prompt}<end_of_turn>
<start_of_turn>model
"""


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _run_phase(model_path: str, picks_file: Path, out_file: Path, label: str) -> None:
    """Load a model, generate completions for every prompt in picks_file, write to out_file."""
    os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')

    import unsloth  # must import first
    from unsloth import FastLanguageModel
    import torch

    # Py3.14 + dill 0.4.0 pickle incompatibility
    import hashlib
    import datasets.fingerprint as _fp
    _fp.Hasher.hash = classmethod(lambda cls, value: hashlib.sha256(repr(value).encode('utf-8', errors='replace')).hexdigest())
    _fp.generate_fingerprint = lambda dataset: hashlib.sha256(repr(id(dataset)).encode()).hexdigest()

    picks = json.loads(picks_file.read_text(encoding='utf-8'))
    print(f'[{label}] model = {model_path}')
    print(f'[{label}] picks = {len(picks)}')

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)
    print(f'[{label}] loaded in {time.time()-t0:.1f}s; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB')

    tok = tokenizer if not hasattr(tokenizer, 'tokenizer') else tokenizer.tokenizer

    results: list[dict] = []
    for i, r in enumerate(picks):
        t = time.time()
        text = _CHAT_TEMPLATE.format(system=_SYSTEM, prompt=r['prompt'])
        ids = tok(text, return_tensors='pt').input_ids.to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                input_ids      = ids,
                max_new_tokens = 400,
                do_sample      = True,
                temperature    = 0.4,
                top_p          = 0.9,
                pad_token_id   = tok.eos_token_id,
            )
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        dt = time.time() - t
        results.append({'prompt': r['prompt'], 'completion': out.strip()})
        print(f'[{label} {i+1}/{len(picks)}] {dt:.1f}s | {out.strip()[:160]}...')

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[{label}] wrote {out_file}')


def _print_side_by_side(picks_file: Path, base_file: Path, adapter_file: Path) -> None:
    picks = json.loads(picks_file.read_text(encoding='utf-8'))
    base = json.loads(base_file.read_text(encoding='utf-8'))
    adapter = json.loads(adapter_file.read_text(encoding='utf-8'))

    print('\n' + '#'*90)
    print('# SIDE-BY-SIDE: BASE vs BRAIN-LoRA')
    print('#'*90)
    for i, (p, b, a) in enumerate(zip(picks, base, adapter)):
        print(f'\n--- sample {i+1}/{len(picks)} ---')
        print(f'PROMPT : {p["prompt"][:300]}...')
        print(f'\nBASE   : {b["completion"][:600]}')
        print(f'\nLoRA   : {a["completion"][:600]}')
        print(f'\nTARGET : {p["completion"][:600]}')


def _orchestrate(args: argparse.Namespace) -> None:
    rows = _load_jsonl(Path(args.dataset))
    rng  = random.Random(args.seed)
    picks = rng.sample(rows, min(args.n, len(rows)))

    work = Path(args.workdir) if args.workdir else Path(os.environ.get('TEMP', '.')) / f'eval_brain_{int(time.time())}'
    work.mkdir(parents=True, exist_ok=True)
    picks_file = work / 'picks.json'
    base_file  = work / 'base_outputs.json'
    adap_file  = work / 'adapter_outputs.json'
    picks_file.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[orch] workdir = {work}')
    print(f'[orch] base    = {BASE_MODEL}')
    print(f'[orch] adapter = {args.adapter}')
    print(f'[orch] samples = {len(picks)} of {len(rows)}')

    py = sys.executable

    # Phase 1: base
    print('\n' + '='*90)
    print('PHASE 1: BASE MODEL (subprocess)')
    print('='*90)
    r = subprocess.run(
        [py, __file__, '--phase', 'base', '--picks-file', str(picks_file), '--out', str(base_file)],
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f'[orch] base phase failed rc={r.returncode}')

    # Phase 2: adapter
    print('\n' + '='*90)
    print(f'PHASE 2: BRAIN LoRA ADAPTER (subprocess) - {args.adapter}')
    print('='*90)
    r = subprocess.run(
        [py, __file__, '--phase', 'adapter',
         '--picks-file', str(picks_file), '--out', str(adap_file),
         '--adapter', args.adapter],
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f'[orch] adapter phase failed rc={r.returncode}')

    _print_side_by_side(picks_file, base_file, adap_file)
    print(f'\n[orch] artifacts in {work}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase',   choices=['base', 'adapter', 'orchestrate'], default='orchestrate')
    ap.add_argument('--adapter', type=str)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--n',       type=int, default=3)
    ap.add_argument('--seed',    type=int, default=2026)
    ap.add_argument('--workdir', type=str, default=None)
    ap.add_argument('--picks-file', type=str)
    ap.add_argument('--out',        type=str)
    args = ap.parse_args()

    if args.phase == 'orchestrate':
        if not args.adapter or not args.dataset:
            raise SystemExit('orchestrate mode needs --adapter and --dataset')
        _orchestrate(args)
    elif args.phase == 'base':
        if not args.picks_file or not args.out:
            raise SystemExit('base phase needs --picks-file and --out')
        _run_phase(BASE_MODEL, Path(args.picks_file), Path(args.out), 'base')
    elif args.phase == 'adapter':
        if not args.picks_file or not args.out or not args.adapter:
            raise SystemExit('adapter phase needs --picks-file, --out, --adapter')
        _run_phase(args.adapter, Path(args.picks_file), Path(args.out), 'lora')


if __name__ == '__main__':
    main()
