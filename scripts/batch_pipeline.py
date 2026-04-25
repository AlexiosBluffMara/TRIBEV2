"""Batch TRIBE v2 + Gemma pipeline over a list of clips.

Runs one long-lived Python process that:
  1. Loads TRIBE v2 once (BF16 + cuDNN SDPA).
  2. Keeps Gemma E4B + 26B warm in Ollama (via keep_alive=60m on each call).
  3. For each clip: gemma-vision → tribe multimodal → narration → artifacts.
  4. Captures hwmon stats per stage per clip into outputs/batch_<ts>/summary.json.

Designed to hold the RTX 5090 at ~75% sustained utilization for hackathon
benchmarking, not for throughput maximization.

Usage:
    python scripts/batch_pipeline.py
    python scripts/batch_pipeline.py assets/dl_nature.mp4 assets/dl_spiderman.mp4
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make bot package importable when launched from scripts/ directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot import config, gemma, media_gate  # noqa: E402
from bot.hwmon import Monitor              # noqa: E402
from bot.pipeline import load_model, run_inference, vram_report  # noqa: E402
from bot.visualize import render_peak_cortex  # noqa: E402


DEFAULT_CLIPS = [
    ROOT / "assets" / "dl_nature.mp4",
    ROOT / "assets" / "dl_neurosci.mp4",
    ROOT / "assets" / "dl_spiderman.mp4",
    ROOT / "assets" / "test_nature_30s.mp4",
    ROOT / "assets" / "test_spiderman_25s.mp4",
    ROOT / "assets" / "test_neurosci_30s.mp4",
]


def _stage(m: Monitor, extras: dict | None = None) -> dict:
    s = m.stats
    out = {
        "duration_s":        round(s.duration_s, 2),
        "gpu_vram_peak_gb":  round(s.gpu_vram_gb_peak, 2),
        "gpu_vram_mean_gb":  round(s.gpu_vram_gb_mean, 2),
        "gpu_util_peak_pct": s.gpu_util_peak,
        "gpu_util_mean_pct": round(s.gpu_util_mean, 1),
        "gpu_temp_peak_c":   s.gpu_temp_peak,
        "gpu_power_peak_w":  round(s.gpu_power_peak, 0),
        "ram_peak_gb":       round(s.ram_used_gb_peak, 1),
    }
    if extras:
        out.update(extras)
    return out


def process_clip(media: Path, out_dir: Path) -> dict:
    slug      = media.stem.lower().replace(" ", "_")[:40]
    clip_dir  = out_dir / slug
    clip_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {"media": str(media), "slug": slug, "stages": {}}

    # Stage A — Gemma multimodal vision (E4B)
    with Monitor(f"[{slug}] gemma-vision") as m:
        cls = media_gate.classify(media, n_frames=4)
    vision = cls.short_description()
    (clip_dir / "gemma_vision.txt").write_text(vision, encoding="utf-8")
    record["stages"]["gemma_vision"] = _stage(m, extras={
        "content_type": cls.content_type,
        "modality":     cls.modality,
    })
    record["vision_summary"] = vision

    # Stage B — TRIBE v2 full multimodal inference
    with Monitor(f"[{slug}] tribe-inference") as m:
        result = run_inference(media)
    record["stages"]["tribe_inference"] = _stage(m, extras={
        "timesteps":    int(result.preds.shape[0]),
        "vertices":     int(result.preds.shape[1]),
        "peak_t_s":     round(result.peak_t / 2.0, 2),
        "top_rois":     result.top_rois[:8],
    })

    # Stage C — cortex peak render
    with Monitor(f"[{slug}] visualize-peak") as m:
        peak_png = render_peak_cortex(result, out_path=clip_dir / "brain_peak.png")
    record["stages"]["visualize_peak"] = _stage(m, extras={"path": str(peak_png)})

    # Stage D — Gemma narration (26B MoE via narrate())
    roi_means = result.roi_df[result.top_rois].abs().mean().to_dict()
    narration_label = f"{slug}. Vision summary: {vision[:300]}"
    with Monitor(f"[{slug}] gemma-narration") as m:
        narration = gemma.narrate(
            top_rois=result.top_rois,
            roi_means=roi_means,
            stimulus_label=narration_label,
            duration_s=result.preds.shape[0] / 2.0,
            peak_time_s=result.peak_t / 2.0,
        )
    (clip_dir / "gemma_narration.txt").write_text(narration, encoding="utf-8")
    record["stages"]["gemma_narration"] = _stage(m, extras={
        "narration_chars": len(narration),
    })
    record["narration"] = narration

    # Persist per-clip record alongside artifacts
    (clip_dir / "record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def main() -> None:
    clips = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_CLIPS
    clips = [c for c in clips if c.exists()]
    if not clips:
        raise SystemExit("No valid clips to process.")

    ts       = int(time.time())
    out_dir  = ROOT / "outputs" / f"batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[batch] out_dir = {out_dir}")
    print(f"[batch] clips   = {len(clips)}")

    # One-time model load: subsequent clips reuse the warm TRIBE process.
    with Monitor("[batch] tribe-warmup") as m:
        load_model()
    warmup_stage = _stage(m, extras=vram_report())

    summary = {
        "ts":       ts,
        "clips":    [str(c) for c in clips],
        "warmup":   warmup_stage,
        "records":  [],
    }

    t0 = time.time()
    for i, clip in enumerate(clips, 1):
        print(f"\n[batch] ({i}/{len(clips)}) {clip.name}")
        try:
            rec = process_clip(clip, out_dir)
            summary["records"].append(rec)
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            err = {"media": str(clip), "error": f"{type(exc).__name__}: {exc}"}
            summary["records"].append(err)
            print(f"[batch] ERROR on {clip.name}: {exc}")

    summary["total_seconds"] = round(time.time() - t0, 2)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    n_ok = sum(1 for r in summary["records"] if "error" not in r)
    print(f"\n[batch] {n_ok}/{len(clips)} clips succeeded in "
          f"{summary['total_seconds']:.1f}s — {out_dir}")


if __name__ == "__main__":
    main()
