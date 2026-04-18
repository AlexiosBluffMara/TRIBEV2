"""End-to-end offline demo: Gemma vision -> TRIBE v2 -> Gemma narration.

Runs on a single media file (video preferred) and prints a hardware strain
report (GPU VRAM, util, RAM) for each stage so we can size the deployment
box. Outputs: outputs/brain_peak.png, outputs/tribev2_stream.mp4,
outputs/gemma_vision.txt, outputs/gemma_narration.txt, outputs/report.json.

Usage:
    python -m bot.run_demo                              # uses the packaged demo clip
    python -m bot.run_demo path/to/your_clip.mp4
    python -m bot.run_demo --skip-stream --skip-vision
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config, gemma, media_gate
from .hwmon import Monitor
from .pipeline import run_inference
from .visualize import render_peak_cortex, render_roi_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path, nargs="?", default=config.DEMO_VIDEO)
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip Gemma's multimodal video description.")
    parser.add_argument("--skip-narration", action="store_true",
                        help="Skip Gemma's brain-response narration.")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--report", type=Path, default=config.OUT_DIR / "report.json")
    args = parser.parse_args()

    media: Path = args.media
    if not media.exists():
        raise SystemExit(f"Media not found: {media}")

    label = args.label or media.stem.replace("_", " ")
    report: dict = {
        "media": str(media),
        "label": label,
        "size_mb": round(media.stat().st_size / 1e6, 2),
        "stages": {},
    }

    print(f"\n[demo] Stimulus: {media} ({report['size_mb']} MB)")
    t_start = time.time()

    # --- Stage 1: Gemma multimodal vision ---
    vision_text = None
    if not args.skip_vision:
        with Monitor("gemma-vision") as m:
            cls = media_gate.classify(media, n_frames=4)
        vision_text = cls.short_description()
        (config.OUT_DIR / "gemma_vision.txt").write_text(vision_text, encoding="utf-8")
        print("\n=== Gemma vision (what's in the media) ===")
        print(vision_text)
        print(f"(content_type: {cls.content_type}, modality: {cls.modality})")
        report["stages"]["gemma_vision"] = _stage(m, extras={
            "frames": [str(f) for f in cls.frames],
            "content_type": cls.content_type,
            "modality": cls.modality,
        })

    # --- Stage 2: TRIBE v2 model load ---
    from .pipeline import load_model
    with Monitor("tribe-load") as m:
        load_model()
    report["stages"]["tribe_load"] = _stage(m)

    # --- Stage 3: TRIBE v2 inference ---
    with Monitor("tribe-inference") as m:
        result = run_inference(media)
    report["stages"]["tribe_inference"] = _stage(m, extras={
        "timesteps": int(result.preds.shape[0]),
        "vertices": int(result.preds.shape[1]),
        "peak_t_s": result.peak_t / 2.0,
    })

    # --- Stage 4: Visualization ---
    with Monitor("visualize-peak") as m:
        peak_png = render_peak_cortex(result)
    report["stages"]["visualize_peak"] = _stage(m, extras={"path": str(peak_png)})

    if not args.skip_stream:
        with Monitor("visualize-stream") as m:
            stream = render_roi_stream(result)
        report["stages"]["visualize_stream"] = _stage(m, extras={"path": str(stream)})

    # --- Stage 5: Gemma narration of the brain response ---
    if not args.skip_narration:
        roi_means = result.roi_df[result.top_rois].abs().mean().to_dict()
        narration_label = (
            f"{label}. Gemma's vision summary: {vision_text[:300]}"
            if vision_text else label
        )
        with Monitor("gemma-narration") as m:
            narration = gemma.narrate(
                top_rois=result.top_rois,
                roi_means=roi_means,
                stimulus_label=narration_label,
                duration_s=result.preds.shape[0] / 2.0,
                peak_time_s=result.peak_t / 2.0,
            )
        (config.OUT_DIR / "gemma_narration.txt").write_text(narration, encoding="utf-8")
        print("\n=== Gemma narration (what the brain is doing) ===")
        print(narration)
        report["stages"]["gemma_narration"] = _stage(m)

    report["total_seconds"] = round(time.time() - t_start, 2)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[demo] Total wall time: {report['total_seconds']}s")
    print(f"[demo] Wrote {args.report}")
    _print_summary(report)


def _stage(monitor: Monitor, extras: dict | None = None) -> dict:
    s = monitor.stats
    d = {
        "duration_s": round(s.duration_s, 2),
        "gpu_vram_peak_gb": round(s.gpu_vram_gb_peak, 2),
        "gpu_vram_mean_gb": round(s.gpu_vram_gb_mean, 2),
        "gpu_util_peak_pct": s.gpu_util_peak,
        "gpu_util_mean_pct": round(s.gpu_util_mean, 1),
        "gpu_temp_peak_c": s.gpu_temp_peak,
        "gpu_power_peak_w": round(s.gpu_power_peak, 0),
        "ram_peak_gb": round(s.ram_used_gb_peak, 1),
    }
    if extras:
        d.update(extras)
    return d


def _print_summary(report: dict) -> None:
    print("\n=== Hardware strain summary ===")
    print(f"{'stage':22s} {'time':>7s} {'VRAM_pk':>9s} {'GPU%_pk':>8s} {'RAM_pk':>8s}")
    for name, s in report["stages"].items():
        print(
            f"{name:22s} "
            f"{s['duration_s']:>6.1f}s "
            f"{s['gpu_vram_peak_gb']:>8.2f}G "
            f"{s['gpu_util_peak_pct']:>7d}% "
            f"{s['ram_peak_gb']:>7.1f}G"
        )


if __name__ == "__main__":
    main()
