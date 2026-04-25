"""Three-way held-out eval: base vs v2-adapter vs v3-adapter, N paired prompts.

Launches llama-server three times sequentially (no --lora, --lora v2, --lora v3),
reuses the same picks.json across all three phases, and writes:
    {out_dir}/picks.json
    {out_dir}/base_outputs.json
    {out_dir}/v2_outputs.json
    {out_dir}/v3_outputs.json

Then snapshots the four files into D:/TRIBEV2/outputs/paper/eval_stats_three_way/<ts>/
and runs compute_three_way_stats.py on the snapshot.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/run_three_way_eval.py \\
        --n 30 --v3-adapter D:/research/weights/gemma3-27b-brain-v3-r64-<ts>/brain-v3-r64-lora-f16.gguf
"""
from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

LLAMA_BIN = Path('D:/research/tmp/llama_bin/llama-server.exe')
BASE_GGUF = Path('D:/research/base_gguf/gemma-3-27b-it-Q4_K_M.gguf')
V2_ADAPTER = Path('D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/brain-v2-r32-lora-f16.gguf')
PORT = 8899
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


def _launch_llama_server(*, lora: Path | None) -> subprocess.Popen:
    cmd = [str(LLAMA_BIN), '-m', str(BASE_GGUF), '-ngl', '99', '-c', '2048',
           '--host', '127.0.0.1', '--port', str(PORT)]
    if lora is not None:
        cmd += ['--lora', str(lora)]
    print(f'[launch] {" ".join(cmd)}')
    creationflags = 0x08000000 if hasattr(subprocess, 'STARTUPINFO') else 0
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=creationflags)


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
        time.sleep(1 if _port_open(PORT) else 2)
    return False


def _kill(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.kill()
        try:
            p.wait(timeout=30)
        except Exception:
            pass
    for _ in range(30):
        if not _port_open(PORT):
            return
        time.sleep(1)


def _run_phase(phase: str, out_dir: Path, n: int, seed: int) -> None:
    cmd = [
        'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe',
        str(REPO_ROOT / 'scripts/eval_brain_llamacpp.py'),
        '--phase', phase, '--n', str(n), '--seed', str(seed),
        '--out-dir', str(out_dir),
    ]
    print(f'[phase {phase}] {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f'phase {phase} failed rc={r.returncode}')


def _run_with_lora(label: str, lora: Path | None, out_dir: Path, n: int, seed: int) -> None:
    p = _launch_llama_server(lora=lora)
    try:
        if not _wait_ready():
            raise SystemExit(f'{label} server never became ready')
        _run_phase(label, out_dir, n, seed)
    finally:
        _kill(p)
        print(f'[{label}] server killed')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--v2-adapter', type=Path, default=V2_ADAPTER)
    ap.add_argument('--v3-adapter', type=Path, required=True,
                    help='Path to brain-v3-r64-lora-f16.gguf')
    ap.add_argument('--skip-base', action='store_true')
    ap.add_argument('--skip-v2', action='store_true')
    ap.add_argument('--skip-v3', action='store_true')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('C:/Users/soumi/AppData/Local/Temp') /
                            f'eval_three_way_{int(time.time())}')
    args = ap.parse_args()

    for need in (LLAMA_BIN, BASE_GGUF, args.v2_adapter, args.v3_adapter):
        if not need.exists():
            raise SystemExit(f'missing required file: {need}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[run] out_dir={args.out_dir}  n={args.n}  seed={args.seed}')
    print(f'[run] v2={args.v2_adapter}')
    print(f'[run] v3={args.v3_adapter}')

    if not args.skip_base:
        _run_with_lora('base', None, args.out_dir, args.n, args.seed)
    if not args.skip_v2:
        _run_with_lora('v2', args.v2_adapter, args.out_dir, args.n, args.seed)
    if not args.skip_v3:
        _run_with_lora('v3', args.v3_adapter, args.out_dir, args.n, args.seed)

    snap = REPO_ROOT / 'outputs/paper/eval_stats_three_way' / args.out_dir.name
    snap.mkdir(parents=True, exist_ok=True)
    for fname in ('picks.json', 'base_outputs.json', 'v2_outputs.json', 'v3_outputs.json'):
        src = args.out_dir / fname
        if src.exists():
            shutil.copy2(src, snap / fname)
    print(f'[run] snapshotted to {snap}')

    stats_cmd = [
        'C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe',
        str(REPO_ROOT / 'scripts/compute_three_way_stats.py'),
        '--snap', str(snap),
    ]
    print(f'[run] stats: {" ".join(stats_cmd)}')
    subprocess.run(stats_cmd)


if __name__ == '__main__':
    main()
