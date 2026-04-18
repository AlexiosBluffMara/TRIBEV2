"""
Google Cloud Storage backend for JemmaBrain results.

When GCS_BUCKET is set in the environment, results are read/written to
gs://{bucket}/results/ instead of (or in addition to) the local filesystem.

This module is a drop-in complement to results_store.py:
  - save_result_gcs()  — upload a completed job to GCS
  - list_results_gcs() — list results from GCS bucket
  - load_result_gcs()  — download a result from GCS
  - sync_from_gcs()    — sync new GCS results to local results dir

Used automatically by server.py when GCS_BUCKET env var is present.
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

# Optional GCS import
try:
    from google.cloud import storage as _gcs
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

GCS_BUCKET = os.getenv('GCS_BUCKET', '').strip()
GCS_RESULTS_PREFIX = 'results/'


def _client():
    """Return a GCS client, or None if unavailable."""
    if not _GCS_AVAILABLE or not GCS_BUCKET:
        return None, None
    try:
        client = _gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        return client, bucket
    except Exception:
        return None, None


# ── Save ──────────────────────────────────────────────────────────────────────

def save_result_gcs(
    job_id: str,
    stimulus_title: str,
    preds: np.ndarray,
    analysis_dict: dict,
    narrations: dict,
    media_filename: str = '',
) -> bool:
    """
    Upload a completed pipeline result to GCS.
    Returns True on success, False if GCS is not configured.
    """
    client, bucket = _client()
    if bucket is None:
        return False

    # Binary BOLD data
    arr = preds.astype(np.float32)
    bold_key = f'{GCS_RESULTS_PREFIX}{job_id}_bold.bin'
    blob = bucket.blob(bold_key)
    blob.upload_from_string(arr.tobytes(), content_type='application/octet-stream')

    # Metadata JSON
    meta = {
        'job_id':         job_id,
        'timestamp':      time.time(),
        'stimulus_title': stimulus_title,
        'media_filename': media_filename,
        'n_trs':          int(preds.shape[0]),
        'n_verts':        int(preds.shape[1]),
        'bold_bin':       f'{job_id}_bold.bin',
        'analysis':       analysis_dict,
        'narrations':     {str(k): v for k, v in narrations.items()},
    }
    meta_key = f'{GCS_RESULTS_PREFIX}{job_id}_meta.json'
    blob = bucket.blob(meta_key)
    blob.upload_from_string(
        json.dumps(meta, indent=2, ensure_ascii=False),
        content_type='application/json',
    )

    return True


# ── List ──────────────────────────────────────────────────────────────────────

def list_results_gcs() -> list[dict]:
    """
    List all results in GCS bucket, newest first.
    Each entry: {job_id, timestamp, stimulus_title, n_trs, n_verts, media_filename}
    """
    client, bucket = _client()
    if bucket is None:
        return []

    results = []
    try:
        blobs = list(client.list_blobs(GCS_BUCKET, prefix=GCS_RESULTS_PREFIX, delimiter='/'))
        meta_blobs = [b for b in blobs if b.name.endswith('_meta.json')]
        for blob in meta_blobs:
            try:
                raw = blob.download_as_text(encoding='utf-8')
                meta = json.loads(raw)
                results.append({
                    'job_id':         meta.get('job_id', ''),
                    'timestamp':      meta.get('timestamp', 0),
                    'stimulus_title': meta.get('stimulus_title', ''),
                    'media_filename': meta.get('media_filename', ''),
                    'n_trs':          meta.get('n_trs', 0),
                    'n_verts':        meta.get('n_verts', 0),
                    'source':         'gcs',
                })
            except Exception:
                pass
    except Exception:
        pass

    return sorted(results, key=lambda x: -x['timestamp'])


# ── Load ──────────────────────────────────────────────────────────────────────

def load_result_gcs(job_id: str) -> dict | None:
    """
    Download and return a result from GCS.
    Returns meta dict with 'bold_data' key (list-of-lists), or None.
    """
    client, bucket = _client()
    if bucket is None:
        return None

    meta_key = f'{GCS_RESULTS_PREFIX}{job_id}_meta.json'
    bold_key = f'{GCS_RESULTS_PREFIX}{job_id}_bold.bin'

    try:
        meta_blob = bucket.blob(meta_key)
        if not meta_blob.exists():
            return None
        meta = json.loads(meta_blob.download_as_text(encoding='utf-8'))

        bold_blob = bucket.blob(bold_key)
        if bold_blob.exists():
            raw  = bold_blob.download_as_bytes()
            arr  = np.frombuffer(raw, dtype=np.float32)
            arr  = arr.reshape(meta['n_trs'], meta['n_verts'])
            meta['bold_data'] = arr.tolist()

        return meta
    except Exception:
        return None


def load_result_bold_bytes_gcs(job_id: str) -> tuple[bytes | None, int, int]:
    """
    Download raw BOLD binary from GCS.
    Returns (raw_bytes, n_trs, n_verts) or (None, 0, 0).
    """
    client, bucket = _client()
    if bucket is None:
        return None, 0, 0

    meta_key = f'{GCS_RESULTS_PREFIX}{job_id}_meta.json'
    bold_key = f'{GCS_RESULTS_PREFIX}{job_id}_bold.bin'

    try:
        meta_blob = bucket.blob(meta_key)
        if not meta_blob.exists():
            return None, 0, 0
        meta = json.loads(meta_blob.download_as_text(encoding='utf-8'))

        bold_blob = bucket.blob(bold_key)
        if not bold_blob.exists():
            return None, 0, 0

        raw = bold_blob.download_as_bytes()
        return raw, meta['n_trs'], meta['n_verts']
    except Exception:
        return None, 0, 0


# ── Sync ──────────────────────────────────────────────────────────────────────

def sync_from_gcs(local_results_dir: Path) -> list[str]:
    """
    Download any GCS results not already on local disk.
    Returns list of newly synced job_ids.
    """
    client, bucket = _client()
    if bucket is None:
        return []

    local_results_dir.mkdir(parents=True, exist_ok=True)
    synced = []

    try:
        blobs = list(client.list_blobs(GCS_BUCKET, prefix=GCS_RESULTS_PREFIX, delimiter='/'))
        meta_blobs = [b for b in blobs if b.name.endswith('_meta.json')]
        for blob in meta_blobs:
            filename = blob.name.split('/')[-1]
            local_path = local_results_dir / filename
            if not local_path.exists():
                blob.download_to_filename(str(local_path))
                # Also download the bold.bin
                job_id = filename.replace('_meta.json', '')
                bold_blob = bucket.blob(f'{GCS_RESULTS_PREFIX}{job_id}_bold.bin')
                local_bold = local_results_dir / f'{job_id}_bold.bin'
                if bold_blob.exists() and not local_bold.exists():
                    bold_blob.download_to_filename(str(local_bold))
                synced.append(job_id)
    except Exception:
        pass

    return synced


# ── Unified list (local + GCS, deduplicated) ──────────────────────────────────

def list_results_unified() -> list[dict]:
    """
    Return results from both local disk and GCS, deduplicated, newest first.
    Used by server.py when GCS_BUCKET is set.
    """
    from . import results_store as _local

    local = {r['job_id']: r for r in _local.list_results()}
    gcs   = {r['job_id']: r for r in list_results_gcs()}

    # Merge: GCS wins on conflict (more authoritative)
    merged = {**local, **gcs}
    return sorted(merged.values(), key=lambda x: -x.get('timestamp', 0))
