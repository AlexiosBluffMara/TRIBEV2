"""
Persistent per-job result storage for the JemmaBrain Three.js viewer.

Each completed pipeline is saved under:
    outputs/results/{job_id}_meta.json   — analysis, narrations, stimulus info
    outputs/results/{job_id}_bold.bin    — raw float32 BOLD preds (n_trs × n_verts)

The Three.js server loads these to replay any past analysis in the browser.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config

RESULTS_DIR = config.OUT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_result(
    job_id: str,
    stimulus_title: str,
    preds: np.ndarray,       # shape (n_trs, n_verts) float32
    analysis_dict: dict,     # from BrainAnalysis.to_dict() or server payload
    narrations: dict,        # {int: str}  tier → text
    media_filename: str = '',
) -> Path:
    """
    Persist a completed pipeline result.  Returns the meta JSON path.
    preds is saved as raw binary float32 (endian = platform default, little on x86).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Binary BOLD data
    bold_path = RESULTS_DIR / f'{job_id}_bold.bin'
    arr = preds.astype(np.float32)
    bold_path.write_bytes(arr.tobytes())

    # Metadata JSON
    meta = {
        'job_id':         job_id,
        'timestamp':      time.time(),
        'stimulus_title': stimulus_title,
        'media_filename': media_filename,
        'n_trs':          int(preds.shape[0]),
        'n_verts':        int(preds.shape[1]),
        'bold_bin':       bold_path.name,
        'analysis':       analysis_dict,
        'narrations':     {str(k): v for k, v in narrations.items()},
    }
    meta_path = RESULTS_DIR / f'{job_id}_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    return meta_path


# ── Load ──────────────────────────────────────────────────────────────────────

def load_result(job_id: str) -> dict | None:
    """
    Load a saved result.  Returns the meta dict with 'bold_data' key added
    (list[list[float]], shape n_trs × n_verts) for the WebSocket/REST endpoint.
    Returns None if not found.
    """
    meta_path = RESULTS_DIR / f'{job_id}_meta.json'
    bold_path = RESULTS_DIR / f'{job_id}_bold.bin'

    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    if bold_path.exists():
        raw  = np.frombuffer(bold_path.read_bytes(), dtype=np.float32)
        arr  = raw.reshape(meta['n_trs'], meta['n_verts'])
        meta['bold_data'] = arr.tolist()

    return meta


def load_result_preds(job_id: str) -> np.ndarray | None:
    """Return raw preds array without converting to list (cheaper for streaming)."""
    meta_path = RESULTS_DIR / f'{job_id}_meta.json'
    bold_path = RESULTS_DIR / f'{job_id}_bold.bin'
    if not meta_path.exists() or not bold_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    raw  = np.frombuffer(bold_path.read_bytes(), dtype=np.float32)
    return raw.reshape(meta['n_trs'], meta['n_verts'])


# ── List ──────────────────────────────────────────────────────────────────────

def list_results() -> list[dict]:
    """
    Return all saved result metadata, newest first.
    Each entry has: job_id, timestamp, stimulus_title, n_trs, n_verts, media_filename.
    """
    results = []
    for p in sorted(RESULTS_DIR.glob('*_meta.json'), key=lambda x: -x.stat().st_mtime):
        try:
            meta = json.loads(p.read_text(encoding='utf-8'))
            results.append({
                'job_id':         meta.get('job_id', p.stem.replace('_meta', '')),
                'timestamp':      meta.get('timestamp', 0),
                'stimulus_title': meta.get('stimulus_title', ''),
                'media_filename': meta.get('media_filename', ''),
                'n_trs':          meta.get('n_trs', 0),
                'n_verts':        meta.get('n_verts', 0),
            })
        except Exception:
            pass
    return results


def latest_result_id() -> str | None:
    """Return the job_id of the most recently saved result."""
    items = list_results()
    return items[0]['job_id'] if items else None
