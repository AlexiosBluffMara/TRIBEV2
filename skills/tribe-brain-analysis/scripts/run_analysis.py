#!/usr/bin/env python3
"""Run TRIBE v2 brain-response analysis on a media file.

Usage:
    python skills/tribe-brain-analysis/scripts/run_analysis.py <media_path> [--brainnetome]

Outputs:
    outputs/brain_peak.png        Peak-frame cortex surface map
    outputs/preds.npy             Raw predictions (T, 20484) float32
    outputs/roi_schaefer400.parq  Schaefer-400 ROI time series
    outputs/report.json           Full structured analysis (JSON)
    stdout                        JSON summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a standalone script
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from bot import analysis as _analysis
from bot import config as _cfg
from bot.pipeline import load_model, run_inference
from bot.visualize import render_peak_cortex


def main() -> None:
    ap = argparse.ArgumentParser(description="TRIBE v2 brain-response analyser")
    ap.add_argument("media_path", help="Path to video/audio/text file (≤50 s)")
    ap.add_argument("--brainnetome", action="store_true",
                    help="Also compute Brainnetome-246 atlas (~50 MB download on first run)")
    ap.add_argument("--high-res", action="store_true",
                    help="Also compute Schaefer-1000 high-resolution atlas (~30 s extra)")
    ap.add_argument("--json-only", action="store_true",
                    help="Print JSON report only; skip PNG rendering")
    args = ap.parse_args()

    media = Path(args.media_path)
    if not media.exists():
        sys.exit(f"Error: file not found: {media}")

    print(f"[run_analysis] Loading TRIBE v2…", flush=True)
    load_model()

    print(f"[run_analysis] Running inference on {media.name}…", flush=True)
    result = run_inference(media)

    print(f"[run_analysis] Running multi-atlas BrainAnalysis…", flush=True)
    ba = _analysis.analyse(
        result,
        high_res=args.high_res,
        harvard_oxford=True,
        juelich=True,
        brainnetome=args.brainnetome,
    )

    # ── Save raw predictions ──────────────────────────────────────────────────
    _cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = _cfg.OUT_DIR / "preds.npy"
    np.save(preds_path, result.preds)
    print(f"[run_analysis] Saved predictions → {preds_path}", flush=True)

    # ── Save Schaefer-400 ROI time series ─────────────────────────────────────
    parq_path = _cfg.OUT_DIR / "roi_schaefer400.parq"
    ba.s400_roi_df.to_parquet(parq_path, index=False)
    print(f"[run_analysis] Saved ROI time series → {parq_path}", flush=True)

    # ── Render cortex PNG ─────────────────────────────────────────────────────
    png_path = None
    if not args.json_only:
        png_path = render_peak_cortex(result)
        print(f"[run_analysis] Rendered cortex PNG → {png_path}", flush=True)

    # ── Build JSON report ─────────────────────────────────────────────────────
    report = ba.to_dict()
    report["media_file"] = str(media)
    report["preds_shape"] = list(result.preds.shape)
    report["seconds_elapsed"] = round(result.seconds_elapsed, 2)
    report["analysis_seconds"] = ba.analysis_seconds
    report["cortex_png"] = str(png_path) if png_path else None
    report["preds_npy"] = str(preds_path)
    report["roi_parquet"] = str(parq_path)

    report_path = _cfg.OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[run_analysis] Saved report → {report_path}", flush=True)

    # ── Print gemma_context (for prompt injection) ────────────────────────────
    print("\n" + "─" * 60)
    print("BRAIN CONTEXT (inject into Gemma 4 prompts):")
    print("─" * 60)
    print(ba.gemma_context())
    print("─" * 60)

    # ── Print JSON summary to stdout ──────────────────────────────────────────
    print("\nJSON SUMMARY:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
