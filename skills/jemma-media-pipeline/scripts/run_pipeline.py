#!/usr/bin/env python3
"""Headless Jemma pipeline — full Gemma 4 + TRIBE v2 analysis without Discord.

Usage:
    python skills/jemma-media-pipeline/scripts/run_pipeline.py <media_path>
    python skills/jemma-media-pipeline/scripts/run_pipeline.py --demo
    python skills/jemma-media-pipeline/scripts/run_pipeline.py <media_path> --all-tiers
    python skills/jemma-media-pipeline/scripts/run_pipeline.py <media_path> --brainnetome

All output goes to outputs/ and stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot import analysis as _analysis
from bot import config, media_gate, tiers
from bot.pipeline import load_model, run_inference, run_inference_text_only
from bot.visualize import render_peak_cortex

# Load .env if present
_env = _ROOT / ".env"
if _env.exists():
    import os
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless Jemma pipeline")
    ap.add_argument("media_path", nargs="?", help="Path to media file")
    ap.add_argument("--demo", action="store_true",
                    help="Use the packaged demo clip")
    ap.add_argument("--all-tiers", action="store_true",
                    help="Generate all 7 expertise tiers (default: tiers 2, 5, 6)")
    ap.add_argument("--brainnetome", action="store_true",
                    help="Include Brainnetome-246 atlas (downloads ~50 MB on first run)")
    args = ap.parse_args()

    if args.demo:
        media = config.DEMO_VIDEO
        if not media.exists():
            sys.exit("Demo clip not found. Run: python -m bot.make_demo_asset")
    elif args.media_path:
        media = Path(args.media_path)
        if not media.exists():
            sys.exit(f"File not found: {media}")
    else:
        ap.print_help()
        sys.exit(1)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"Jemma pipeline — {media.name}")
    print(f"{'─'*60}\n")

    # ── Stage A: Gemma 4 vision ──────────────────────────────────────────────
    print("▶ Stage A: Gemma 4 vision description…", flush=True)
    cls = media_gate.classify(media)
    print(f"  Content type : {cls.content_type}")
    print(f"  Subject      : {cls.subject}")
    print(f"  Modality     : {cls.modality}")
    print(f"  Description  : {cls.description[:120]}…\n")
    (config.OUT_DIR / "gemma_vision.txt").write_text(cls.description, encoding="utf-8")

    # ── Stage B: text-only TRIBE ─────────────────────────────────────────────
    print("▶ Stage B: text-only TRIBE fast path…", flush=True)
    quick_result = run_inference_text_only(cls.short_description())
    quick_text = tiers.narrate_quick(quick_result, cls.short_description())
    print(f"  Quick read: {quick_text[:200]}…\n")

    # ── Stage C: full multimodal TRIBE ───────────────────────────────────────
    print("▶ Stage C: full multimodal TRIBE v2 (this takes ~4-7 min)…", flush=True)
    result = run_inference(media)
    full_secs = time.time() - t0
    print(f"  Shape : {result.preds.shape}")
    print(f"  Peak  : t={result.peak_t / 2:.1f}s")
    print(f"  Elapsed: {full_secs:.0f}s\n")

    # ── BrainAnalysis ────────────────────────────────────────────────────────
    print("▶ BrainAnalysis: multi-atlas projection…", flush=True)
    ba = _analysis.analyse(
        result,
        harvard_oxford=True,
        juelich=True,
        brainnetome=args.brainnetome,
    )
    brain_ctx = ba.gemma_context()
    print(f"  Dominant network : {_analysis._YEO7_FULL.get(ba.dominant_network, ba.dominant_network)}")
    print(f"  Activated        : {ba.vertices_above_1sd:,} / 20,484 vertices above 1σ "
          f"({ba.activation_fraction_1sd*100:.1f}%)\n")

    # ── Render cortex PNG ────────────────────────────────────────────────────
    print("▶ Rendering peak cortex PNG…", flush=True)
    peak_png = render_peak_cortex(result)
    print(f"  Saved: {peak_png}\n")

    # ── Tier narrations ──────────────────────────────────────────────────────
    label = f"{media.name} — {cls.short_description()}"
    narrations: dict[int, str] = {}

    if args.all_tiers:
        print("▶ Generating all 7 expertise tiers…", flush=True)
        for tier_idx in range(7):
            print(f"  Tier {tier_idx}…", end=" ", flush=True)
            narrations[tier_idx] = tiers.narrate_tier(result, label, tier_idx, brain_ctx)
            print(f"({len(narrations[tier_idx])} chars)")
    else:
        print("▶ Generating 3-tier narration (tiers 2, 5, 6)…", flush=True)
        narr = tiers.narrate_tiered(result, label, brain_ctx)
        narrations[2] = narr.layperson
        narrations[5] = narr.clinician
        narrations[6] = narr.researcher
        print(f"  Done.\n")

    # ── Save outputs ─────────────────────────────────────────────────────────
    narr_text = "\n\n".join(
        f"=== TIER {t} ===\n{txt}" for t, txt in sorted(narrations.items())
    )
    (config.OUT_DIR / "gemma_narration.txt").write_text(narr_text, encoding="utf-8")

    report = ba.to_dict()
    report["media_file"] = str(media)
    report["preds_shape"] = list(result.preds.shape)
    report["pipeline_seconds"] = round(time.time() - t0, 1)
    report["narrations"] = {str(k): v for k, v in narrations.items()}
    report["gemma_vision"] = cls.description
    report["quick_read"] = quick_text
    (config.OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"✅ Done in {(time.time() - t0)/60:.1f} min")
    print(f"   Cortex PNG : {peak_png}")
    print(f"   Report     : {config.OUT_DIR / 'report.json'}")
    print(f"   Narrations : {config.OUT_DIR / 'gemma_narration.txt'}")
    print(f"{'─'*60}\n")

    for tier_idx, txt in sorted(narrations.items()):
        tier_labels = {0:"Toddler",1:"General adult",2:"Curious adult",
                       3:"High school",4:"College",5:"Clinician",6:"Researcher"}
        print(f"\n{'━'*60}")
        print(f"Tier {tier_idx}: {tier_labels.get(tier_idx, '?')}")
        print(f"{'━'*60}")
        print(txt)


if __name__ == "__main__":
    main()
