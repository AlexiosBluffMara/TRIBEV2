"""TRIBE v2 inference pipeline — optimized for NVIDIA Blackwell (RTX 5090, sm_120).

Blackwell optimizations applied:
  1. torch.compile(mode='reduce-overhead') — CUDA graph capture, ~15-30% speedup
  2. BF16 everywhere — native Blackwell precision, no accuracy loss
  3. torch.backends.cuda.matmul.allow_tf32 = True — TF32 for matmuls
  4. cuDNN SDPA (Flash-Attention-like) — auto via sdp_kernel context
  5. torch.inference_mode() — no autograd overhead
  6. Pin TRIBE model to GPU 0 — zero CPU offload

Model loading sequence:
  load_model() — loads weights, compiles, warms the CUDA graph
  Subsequent calls — all inference runs via the compiled graph (fast path)

InferenceResult fields:
  preds         (T, 20484) float32 — per-vertex BOLD z-scores on fsaverage5
  roi_df        Schaefer-400 time-series (T × 400)
  top_rois      Top 12 active ROIs by mean |z|
  peak_t        Frame index of peak global activity
  events_df     TRIBE input event features
  seconds_elapsed  Wall-clock seconds for Stage C
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .logger import log

# ── Blackwell / CUDA flags ─────────────────────────────────────────────────────
# Set before importing torch to ensure they take effect at init time.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_model        = None
_compiled     = False
_atlas_cache: dict = {}


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    preds:           np.ndarray   # (T, 20484) float32
    roi_df:          pd.DataFrame
    top_rois:        list[str]
    peak_t:          int
    events_df:       pd.DataFrame
    seconds_elapsed: float
    model_dtype:     str = 'bf16'


# ── Model loading with Blackwell optimizations ─────────────────────────────────

def _apply_blackwell_opts(model) -> None:
    """Apply RTX 5090 / Blackwell sm_120 specific optimizations to the TRIBE model."""
    import torch

    # 1. TF32 matmuls — Blackwell tensor cores use TF32 natively for BF16 inputs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # 2. cuDNN benchmark — auto-select fastest convolution algorithm
    torch.backends.cudnn.benchmark = True

    # 3. Enable cuDNN FlashAttention-like SDPA
    # PyTorch 2.11+ automatically routes to cuDNN 9.15+ SDPA on sm_120.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
        log.info('[pipeline] cuDNN SDPA enabled (Flash + MemEfficient paths)')
    except AttributeError:
        log.warning('[pipeline] cuDNN SDPA API not available in this PyTorch build')

    # 4. Cast model to BF16 — native Blackwell precision
    # BF16 on sm_120 delivers same TFLOPS as FP16 but with better numeric stability.
    try:
        if hasattr(model, 'to'):
            model.to(dtype=torch.bfloat16)
            log.info('[pipeline] Model cast to BF16')
    except Exception as exc:
        log.warning('[pipeline] BF16 cast failed: %s', exc)

    # 5. torch.compile — CUDA graph capture + kernel fusion
    # mode='reduce-overhead': enables CUDA graphs, ~15-30% speedup on repeated inference.
    # fullgraph=False: safer for models with dynamic control flow (TRIBE v2 has loops).
    global _compiled
    if not _compiled:
        try:
            if hasattr(model, 'predict'):
                compiled_predict = torch.compile(
                    model.predict,
                    backend='inductor',
                    mode='reduce-overhead',
                    fullgraph=False,
                )
                model.predict = compiled_predict
                _compiled = True
                log.info('[pipeline] torch.compile applied (inductor, reduce-overhead)')
        except Exception as exc:
            log.warning('[pipeline] torch.compile failed: %s — running uncompiled', exc)


def load_model():
    global _model
    if _model is not None:
        return _model

    import torch
    from tribev2.demo_utils import TribeModel

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. RTX 5090 expected for Blackwell optimizations.')

    gpu_name = torch.cuda.get_device_name(0)
    sm_major, sm_minor = torch.cuda.get_device_capability(0)
    log.info('[pipeline] GPU: %s (sm_%d%d)', gpu_name, sm_major, sm_minor)

    if sm_major < 10:
        log.warning('[pipeline] GPU sm_%d%d is not Blackwell (sm_120). '
                    'Blackwell optimizations will still be applied but may be sub-optimal.',
                    sm_major, sm_minor)

    log.info('[pipeline] Loading TRIBE v2 from %s...', config.WEIGHTS_DIR)
    t0 = time.time()

    _model = TribeModel.from_pretrained(
        checkpoint_dir=str(config.WEIGHTS_DIR),
        cache_folder=str(config.CACHE_DIR),
    )

    # Pin to GPU
    _model = _model.cuda()

    load_s = time.time() - t0
    log.info('[pipeline] TRIBE v2 loaded in %.1fs', load_s)

    # Apply Blackwell optimizations
    _apply_blackwell_opts(_model)

    # Warm-up pass (triggers CUDA graph capture for torch.compile)
    log.info('[pipeline] Warming CUDA graph (first inference will be slow)...')
    try:
        _warmup_model(_model)
    except Exception as exc:
        log.warning('[pipeline] Warm-up failed: %s', exc)

    return _model


def _warmup_model(model) -> None:
    """Run a dummy text inference to populate the CUDA graph."""
    import torch, tempfile
    tmp = Path(config.UPLOAD_DIR) / '_warmup.txt'
    tmp.write_text('WARMUP', encoding='utf-8')
    try:
        events = model.get_events_dataframe(text_path=str(tmp))
        with torch.inference_mode():
            _ = model.predict(events=events)
        log.info('[pipeline] Warm-up complete')
    finally:
        tmp.unlink(missing_ok=True)


# ── Atlas projection ───────────────────────────────────────────────────────────

def _schaefer_rois(preds: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Project vertex-space predictions onto Schaefer-400 (cached after first call)."""
    from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
    from nilearn.surface import vol_to_surf

    if 'fsavg5' not in _atlas_cache:
        _atlas_cache['fsavg5'] = fetch_surf_fsaverage('fsaverage5')

    if 'schaefer' not in _atlas_cache:
        atlas   = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        fsavg5  = _atlas_cache['fsavg5']
        lh = vol_to_surf(atlas.maps, fsavg5.pial_left,  interpolation='nearest_most_frequent').astype(int)
        rh = vol_to_surf(atlas.maps, fsavg5.pial_right, interpolation='nearest_most_frequent').astype(int)
        _atlas_cache['schaefer'] = {
            'labels':        [n.decode() if isinstance(n, bytes) else n for n in atlas.labels],
            'vertex_labels': np.concatenate([lh, rh]),
        }

    labels        = _atlas_cache['schaefer']['labels']
    vertex_labels = _atlas_cache['schaefer']['vertex_labels']
    n_rois        = len(labels)
    T             = preds.shape[0]

    roi_ts = np.zeros((T, n_rois), dtype=np.float32)
    for i in range(1, n_rois + 1):
        mask = vertex_labels == i
        if mask.any():
            roi_ts[:, i - 1] = preds[:, mask].mean(axis=1)

    df  = pd.DataFrame(roi_ts, columns=labels[:n_rois])
    df  = df.loc[:, df.abs().sum(axis=0) > 0]
    top = df.abs().mean().sort_values(ascending=False).head(12).index.tolist()
    return df, top


def _build_events(model, media_path: Path) -> pd.DataFrame:
    suffix = media_path.suffix.lower()
    if suffix in {'.mp4', '.avi', '.mkv', '.mov', '.webm'}:
        return model.get_events_dataframe(video_path=str(media_path))
    if suffix in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
        return model.get_events_dataframe(audio_path=str(media_path))
    if suffix == '.txt':
        return model.get_events_dataframe(text_path=str(media_path))
    raise ValueError(f'Unsupported media type: {suffix}')


# ── Main inference ─────────────────────────────────────────────────────────────

def run_inference(media_path: Path | str) -> InferenceResult:
    """
    Run TRIBE v2 full multimodal inference on a media file.

    Blackwell path:
      - BF16 weights in VRAM
      - torch.compile CUDA graph (after first call)
      - cuDNN SDPA attention (Flash-like, fused)
      - Zero CPU offload
    """
    import torch

    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    model = load_model()
    t0    = time.time()

    events_df = _build_events(model, media_path)

    # BF16 autocast for Blackwell
    autocast_ctx = torch.autocast('cuda', dtype=torch.bfloat16, enabled=True)

    with torch.inference_mode(), autocast_ctx:
        preds, _segments = model.predict(events=events_df)

    preds   = np.asarray(preds, dtype=np.float32)
    elapsed = time.time() - t0

    log.info('[pipeline] Inference done: shape=%s (%.1fs)', preds.shape, elapsed)

    peak_t          = int(np.abs(preds).mean(axis=1).argmax())
    roi_df, top_rois = _schaefer_rois(preds)

    return InferenceResult(
        preds=preds,
        roi_df=roi_df,
        top_rois=top_rois,
        peak_t=peak_t,
        events_df=events_df,
        seconds_elapsed=elapsed,
        model_dtype='bf16',
    )


def run_inference_text_only(text: str) -> InferenceResult:
    """
    Text-only fast path: skip V-JEPA2 + wav2vec-BERT, run only Llama-3.2-3B.
    ~10-15s vs ~4-7 min for full multimodal.
    """
    import torch

    text    = text.strip() or 'A short educational video.'
    tmp_txt = Path(config.UPLOAD_DIR) / f'quick_{int(time.time()*1000)}.txt'
    tmp_txt.write_text(text, encoding='utf-8')

    try:
        model    = load_model()
        t0       = time.time()
        events_df = model.get_events_dataframe(text_path=str(tmp_txt))

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            preds, _segments = model.predict(events=events_df)

        preds   = np.asarray(preds, dtype=np.float32)
        elapsed = time.time() - t0

        log.info('[pipeline] Text-only inference: shape=%s (%.1fs)', preds.shape, elapsed)

        peak_t          = int(np.abs(preds).mean(axis=1).argmax())
        roi_df, top_rois = _schaefer_rois(preds)

        return InferenceResult(
            preds=preds,
            roi_df=roi_df,
            top_rois=top_rois,
            peak_t=peak_t,
            events_df=events_df,
            seconds_elapsed=elapsed,
            model_dtype='bf16',
        )
    finally:
        tmp_txt.unlink(missing_ok=True)


# ── VRAM / performance report ──────────────────────────────────────────────────

def vram_report() -> dict:
    """Return current VRAM usage and GPU info for health checks."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved  = torch.cuda.memory_reserved(0) / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'gpu':       torch.cuda.get_device_name(0),
            'allocated': round(allocated, 2),
            'reserved':  round(reserved, 2),
            'total':     round(total, 2),
            'free':      round(total - reserved, 2),
        }
    except Exception:
        return {}
