"""TRIBE v2 inference pipeline.

Loads the model once per process, accepts a media path (video/audio/text)
and returns predicted BOLD activity on the fsaverage5 cortical surface plus
a Schaefer-400 ROI aggregate time-series.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_model = None
_atlas_cache: dict = {}


@dataclass
class InferenceResult:
    preds: np.ndarray
    roi_df: pd.DataFrame
    top_rois: list[str]
    peak_t: int
    events_df: pd.DataFrame
    seconds_elapsed: float


def load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from tribev2.demo_utils import TribeModel

    assert torch.cuda.is_available(), "CUDA GPU required (expected RTX 5090)."
    print(f"[pipeline] Loading TRIBE v2 from {config.WEIGHTS_DIR}...")
    t0 = time.time()
    _model = TribeModel.from_pretrained(
        checkpoint_dir=str(config.WEIGHTS_DIR),
        cache_folder=str(config.CACHE_DIR),
    )
    print(f"[pipeline] TRIBE v2 loaded in {time.time()-t0:.1f}s")
    return _model


def _build_events(model, media_path: Path) -> pd.DataFrame:
    suffix = media_path.suffix.lower()
    if suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        return model.get_events_dataframe(video_path=str(media_path))
    if suffix in {".wav", ".mp3", ".flac", ".ogg"}:
        return model.get_events_dataframe(audio_path=str(media_path))
    if suffix == ".txt":
        return model.get_events_dataframe(text_path=str(media_path))
    raise ValueError(f"Unsupported media type: {suffix}")


def _schaefer_rois(preds: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Project vertex-space predictions onto the Schaefer-400 atlas."""
    from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
    from nilearn.surface import vol_to_surf

    if "fsavg5" not in _atlas_cache:
        _atlas_cache["fsavg5"] = fetch_surf_fsaverage("fsaverage5")
    if "schaefer" not in _atlas_cache:
        atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        fsavg5 = _atlas_cache["fsavg5"]
        lh = vol_to_surf(atlas.maps, fsavg5.pial_left, interpolation="nearest_most_frequent").astype(int)
        rh = vol_to_surf(atlas.maps, fsavg5.pial_right, interpolation="nearest_most_frequent").astype(int)
        _atlas_cache["schaefer"] = {
            "labels": [n.decode() if isinstance(n, bytes) else n for n in atlas.labels],
            "vertex_labels": np.concatenate([lh, rh]),
        }
    labels = _atlas_cache["schaefer"]["labels"]
    vertex_labels = _atlas_cache["schaefer"]["vertex_labels"]

    n_rois = len(labels)
    T = preds.shape[0]
    roi_ts = np.zeros((T, n_rois))
    for i in range(1, n_rois + 1):
        mask = vertex_labels == i
        if mask.any():
            roi_ts[:, i - 1] = preds[:, mask].mean(axis=1)
    df = pd.DataFrame(roi_ts, columns=labels[:n_rois])
    df = df.loc[:, df.abs().sum(axis=0) > 0]
    top = df.abs().mean().sort_values(ascending=False).head(12).index.tolist()
    return df, top


def run_inference(media_path: Path | str) -> InferenceResult:
    """Run TRIBE v2 on a single media file and package the results."""
    import torch

    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    model = load_model()
    t0 = time.time()
    events_df = _build_events(model, media_path)
    with torch.inference_mode():
        preds, _segments = model.predict(events=events_df)
    preds = np.asarray(preds)
    elapsed = time.time() - t0
    print(f"[pipeline] Inference done: shape={preds.shape} ({elapsed:.1f}s)")

    peak_t = int(np.abs(preds).mean(axis=1).argmax())
    roi_df, top_rois = _schaefer_rois(preds)

    return InferenceResult(
        preds=preds,
        roi_df=roi_df,
        top_rois=top_rois,
        peak_t=peak_t,
        events_df=events_df,
        seconds_elapsed=elapsed,
    )


def run_inference_text_only(text: str) -> InferenceResult:
    """Text-only fast path: feed Gemma's description to TRIBE as a .txt stimulus.

    Much faster than video (~10-15 s instead of 7 min) because V-JEPA2 and
    wav2vec-BERT are skipped — only the Llama-3.2-3B text-feature extractor
    runs. The returned prediction reflects language/semantic cortex response
    to hearing/reading the description, not the visual response to the clip.
    """
    import torch

    text = text.strip() or "A short video."
    tmp_txt = config.UPLOAD_DIR / f"quick_{int(time.time()*1000)}.txt"
    tmp_txt.write_text(text, encoding="utf-8")
    try:
        model = load_model()
        t0 = time.time()
        events_df = model.get_events_dataframe(text_path=str(tmp_txt))
        with torch.inference_mode():
            preds, _segments = model.predict(events=events_df)
        preds = np.asarray(preds)
        elapsed = time.time() - t0
        print(f"[pipeline] Text-only inference done: shape={preds.shape} ({elapsed:.1f}s)")
        peak_t = int(np.abs(preds).mean(axis=1).argmax())
        roi_df, top_rois = _schaefer_rois(preds)
        return InferenceResult(
            preds=preds, roi_df=roi_df, top_rois=top_rois,
            peak_t=peak_t, events_df=events_df, seconds_elapsed=elapsed,
        )
    finally:
        tmp_txt.unlink(missing_ok=True)
