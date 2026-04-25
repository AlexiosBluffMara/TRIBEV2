"""GPU health + OOM/network-error signature detection.

Thin wrappers around nvidia-smi so the scheduler can block until enough
VRAM is free, plus log-tail scanners so self-healing can distinguish
OOM from HF-download stalls from garden-variety crashes.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

OOM_SIGNATURES = (
    'CUDA out of memory',
    'torch.cuda.OutOfMemoryError',
    'RuntimeError: out of memory',
    'CUBLAS_STATUS_ALLOC_FAILED',
)

HF_DOWNLOAD_SIGNATURES = (
    'ConnectionError',
    'ReadTimeout',
    'ReadTimeoutError',
    'LocalEntryNotFoundError',
    'IncompleteRead',
    'huggingface_hub.utils._errors.HfHubHTTPError',
)

TRANSIENT_SIGNATURES = (
    'CUDNN_STATUS_EXECUTION_FAILED',
    'NCCL error',
)


def _nvidia_smi_one(field: str) -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,noheader,nounits'],
            creationflags=CREATE_NO_WINDOW, text=True, timeout=10,
        )
    except Exception:
        return ''
    return out.strip().split('\n', 1)[0].strip()


def gpu_free_gb() -> float:
    s = _nvidia_smi_one('memory.free')
    try:
        return float(s) / 1024.0
    except ValueError:
        return 0.0


def gpu_used_gb() -> float:
    s = _nvidia_smi_one('memory.used')
    try:
        return float(s) / 1024.0
    except ValueError:
        return 0.0


def gpu_utilization() -> float:
    s = _nvidia_smi_one('utilization.gpu')
    try:
        return float(s)
    except ValueError:
        return 0.0


def gpu_temperature_c() -> float:
    s = _nvidia_smi_one('temperature.gpu')
    try:
        return float(s)
    except ValueError:
        return 0.0


def gpu_snapshot() -> dict:
    return {
        'free_gb': gpu_free_gb(),
        'used_gb': gpu_used_gb(),
        'util_pct': gpu_utilization(),
        'temp_c': gpu_temperature_c(),
    }


def wait_for_gpu(min_free_gb: float, max_wait_min: int = 30,
                 poll_s: int = 5, log: Callable[[str], None] | None = None,
                 log_every_s: int = 30) -> bool:
    """Block until ≥ `min_free_gb` VRAM is free. Return False on timeout.

    Polls every `poll_s` seconds; rate-limits log output to `log_every_s`.
    If `min_free_gb` is 0, returns True immediately (CPU task).
    """
    if min_free_gb <= 0:
        return True
    t0 = time.time()
    last_log = 0.0
    while time.time() - t0 < max_wait_min * 60:
        free = gpu_free_gb()
        used = gpu_used_gb()
        if free >= min_free_gb:
            if log:
                log(f'[gpu] ready (free={free:.1f}GB, used={used:.1f}GB, '
                    f'waited={time.time()-t0:.0f}s)')
            return True
        now = time.time()
        if log and now - last_log >= log_every_s:
            log(f'[gpu] waiting (free={free:.1f}GB, need={min_free_gb:.1f}GB, '
                f'used={used:.1f}GB)')
            last_log = now
        time.sleep(poll_s)
    return False


def detect_error_signatures(log_path: Path, tail_bytes: int = 32768) -> dict:
    """Scan the tail of a log file for known fault signatures.

    Returns {'oom': bool, 'hf_download': bool, 'transient': bool,
             'tail': <last ~2KB of log>}. All False on missing file.
    """
    empty = {'oom': False, 'hf_download': False, 'transient': False, 'tail': ''}
    if not log_path.exists():
        return empty
    try:
        with log_path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            raw = f.read()
    except OSError:
        return empty
    text = raw.decode('utf-8', errors='replace')
    return {
        'oom': any(s in text for s in OOM_SIGNATURES),
        'hf_download': any(s in text for s in HF_DOWNLOAD_SIGNATURES),
        'transient': any(s in text for s in TRANSIENT_SIGNATURES),
        'tail': text[-2000:],
    }
