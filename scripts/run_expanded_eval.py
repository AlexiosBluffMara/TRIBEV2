"""Orchestrate N=30 held-out eval: launch llama-server for base then adapter,
run generations, tear down, and compute stats.

Blocks on VRAM — synth-gen must be stopped first. This script will NOT kill
running ollama; the user is responsible for freeing VRAM before running.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/run_expanded_eval.py --n 30

Writes eval to C:/Users/soumi/AppData/Local/Temp/eval_brain_llamacpp_<ts>/ and copies a
snapshot of the three JSONs into D:/TRIBEV2/outputs/paper/eval_stats_n30/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

LLAMA_BIN = Path('D:/research/tmp/llama_bin/llama-server.exe')
BASE_GGUF = Path('D:/research/base_gguf/gemma-3-27b-it-Q4_K_M.gguf')
ADAPTER   = Path('D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/brain-v2-r32-lora-f16.gguf')
PORT      = 8899
HEALTH_URL = f'http://127.0.0.1:{PORT}/health'
REPO_ROOT = Path('D:/TRIBEV2')


def _port_open(p: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect(('127.0.0.1', p))
            return True
        except Exception:
            return False


def _launch_llama_server(*, with_adapter: bool) -> subprocess.Popen:
    cmd = [str(LLAMA_BIN), '-m', str(BASE_GGUF), '-ngl', '99', '-c', '2048',
           '--host', '127.0.0.1', '--port', str(PORT)]
    if with_adapter:
        cmd += ['--lora', str(ADAPTER)]
    print(f'[launch] {" ".join(cmd)}')
    # Windows: CREATE_NO_WINDOW so the server doesn't pop a console
    creationflags = 0x08000000 if hasattr(subprocess, 'STARTUPINFO') else 0
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=creationflags)
    return p


def _wait_ready(timeout_s: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    print(f'[launch] ready after {time.time()-t0:.1f}s')
                    return True
        except Exception:
            pass
        if not _port_open(PORT):
            time.sleep(2)
            continue
        time.sleep(1)
    return False


def _kill(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.kill()
        try:
            p.wait(timeout=30)
        except Exception:
            pass
    # wait for port to close
    for _ in range(30):
        if not _port_open(PORT):
            return
        time.sleep(1)


def _run_phase(phase: str, out_dir: Path, n: int, seed: int) -> None:
    cmd = [
        'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe',
        str(REPO_ROOT / 'scripts/eval_brain_llamacpp.py'),
        '--phase', phase,
        '--n', str(n),
        '--seed', str(seed),
        '--out-dir', str(out_dir),
    ]
    print(f'[phase {phase}] {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f'phase {phase} failed rc={r.returncode}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--skip-base', action='store_true')
    ap.add_argument('--skip-adapter', action='store_true')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('C:/Users/soumi/AppData/Local/Temp') /
                            f'eval_brain_llamacpp_{int(time.time())}')
    args = ap.parse_args()

    for need in (LLAMA_BIN, BASE_GGUF, ADAPTER):
        if not need.exists():
            raise SystemExit(f'missing required file: {need}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[run] out_dir={args.out_dir}  n={args.n}  seed={args.seed}')

    if not args.skip_base:
        p = _launch_llama_server(with_adapter=False)
        try:
            if not _wait_ready():
                raise SystemExit('base server never became ready')
            _run_phase('base', args.out_dir, args.n, args.seed)
        finally:
            _kill(p)
            print('[base] server killed')

    if not args.skip_adapter:
        p = _launch_llama_server(with_adapter=True)
        try:
            if not _wait_ready():
                raise SystemExit('adapter server never became ready')
            _run_phase('adapter', args.out_dir, args.n, args.seed)
        finally:
            _kill(p)
            print('[adapter] server killed')

    # Copy outputs into the repo's paper dir for git tracking
    snap = REPO_ROOT / 'outputs/paper/eval_stats_n30' / args.out_dir.name
    snap.mkdir(parents=True, exist_ok=True)
    for fname in ('picks.json', 'base_outputs.json', 'adapter_outputs.json'):
        src = args.out_dir / fname
        if src.exists():
            shutil.copy2(src, snap / fname)
    print(f'[run] snapshotted to {snap}')

    # Run stats — update EVAL_DIRS in compute_eval_stats.py-style by building a one-off
    stats_cmd = [
        'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe', '-c',
        f"import sys, runpy; sys.path.insert(0, r'{REPO_ROOT / 'scripts'}'); "
        f"import compute_eval_stats as m; m.EVAL_DIRS=[{str(args.out_dir)!r}]; "
        f"m.EVAL_DIRS=[__import__('pathlib').Path({str(args.out_dir)!r})]; "
        f"m.OUT_DIR=__import__('pathlib').Path(r'{REPO_ROOT / 'outputs/paper/eval_stats_n30'}'); "
        f"m.OUT_DIR.mkdir(parents=True, exist_ok=True); m.main()",
    ]
    print(f'[run] stats: {" ".join(stats_cmd)}')
    subprocess.run(stats_cmd)


if __name__ == '__main__':
    main()
