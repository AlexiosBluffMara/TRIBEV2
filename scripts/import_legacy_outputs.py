"""
Import the existing outputs/ directory into the results store so all
previous analyses become viewable in the Three.js browser.

The legacy layout (from previous bot runs) is:
    outputs/preds.npy            — TRIBE v2 predictions (100 × 20484)
    outputs/report.json          — analysis dict
    outputs/gemma_narration.txt  — tier 2 narration
    outputs/stimulus.txt         — stimulus name/label

This script copies them into:
    outputs/results/{job_id}_bold.bin
    outputs/results/{job_id}_meta.json

Usage:
    python scripts/import_legacy_outputs.py [--title "My Video"]

After running, open http://localhost:5173 and you'll see the past analysis
in the gallery picker, or link directly with /?r=<job_id>.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'bot'))

OUT_DIR     = ROOT / 'outputs'
RESULTS_DIR = OUT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def import_legacy():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', default='', help='Override stimulus title')
    ap.add_argument('--job-id', default='', help='Override job ID (default: auto-generated)')
    args = ap.parse_args()

    preds_path  = OUT_DIR / 'preds.npy'
    report_path = OUT_DIR / 'report.json'
    narr_path   = OUT_DIR / 'gemma_narration.txt'
    stim_path   = OUT_DIR / 'stimulus.txt'

    if not preds_path.exists():
        print(f'ERROR: {preds_path} not found.')
        print('Make sure you have a completed pipeline run in outputs/.')
        sys.exit(1)

    preds = np.load(str(preds_path)).astype(np.float32)
    print(f'Loaded preds: {preds.shape}  ({preds.nbytes / 1e6:.1f} MB)')

    # Stimulus title
    stimulus_title = args.title
    if not stimulus_title and stim_path.exists():
        stimulus_title = stim_path.read_text(encoding='utf-8').strip()
    if not stimulus_title:
        stimulus_title = 'Legacy analysis'

    # Narration
    narrations = {}
    if narr_path.exists():
        narrations[2] = narr_path.read_text(encoding='utf-8').strip()

    # Analysis dict
    analysis_dict = {}
    if report_path.exists():
        try:
            raw = json.loads(report_path.read_text(encoding='utf-8'))
            # Map legacy keys to the server's expected format
            analysis_dict = raw.get('stages', raw)
        except Exception as e:
            print(f'Warning: could not parse report.json: {e}')

    # Job ID
    job_id = args.job_id or f'legacy_{int(time.time())}'

    # Save — import as package from project root
    import importlib, sys as _sys
    _sys.path.insert(0, str(ROOT))
    # results_store uses relative imports from inside the bot package
    # so we import it via the package path
    import bot.results_store as _rs
    meta_path = _rs.save_result(
        job_id=job_id,
        stimulus_title=stimulus_title,
        preds=preds,
        analysis_dict=analysis_dict,
        narrations=narrations,
        media_filename=stimulus_title,
    )

    print(f'\nImported as job_id: {job_id}')
    print(f'Meta: {meta_path}')
    print(f'Bold: {RESULTS_DIR / f"{job_id}_bold.bin"}')
    print()

    from bot.tunnel import get_public_url, viewer_url  # noqa
    base = get_public_url()
    url  = viewer_url(job_id, base)
    print(f'3D viewer URL: {url}')
    if not base:
        print('(local only — run start_tunnel.py to get a public URL)')
    print()
    print('Open the URL in your browser while `python start_viz.py` is running.')


if __name__ == '__main__':
    import_legacy()
