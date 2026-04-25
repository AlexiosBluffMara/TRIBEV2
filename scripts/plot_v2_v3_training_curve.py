"""Plot v2 + v3 training loss curves side by side.

Reads trainer_state.json from each run's checkpoint-<final_step> dir (auto-detected
via `glob` under each run dir), plots the two loss curves on a shared axis, and emits
loss_curve_v2_v3.{png,svg} + loss_summary_v2_v3.json to D:/TRIBEV2/outputs/paper/figures/.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_v2_v3_training_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V2_RUN = Path('D:/research/weights/gemma3-27b-brain-v2-r32-1776635086')
V3_GLOB = 'D:/research/weights/gemma3-27b-brain-v3-r64-*'
OUT_DIR = Path('D:/TRIBEV2/outputs/paper/figures')


def _latest_v3_run() -> Path | None:
    from glob import glob
    cands = sorted((Path(p) for p in glob(V3_GLOB)), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _final_trainer_state(run: Path) -> Path | None:
    ckpts = sorted((p for p in run.iterdir() if p.is_dir() and p.name.startswith('checkpoint-')),
                   key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    for c in ckpts:
        if (c / 'trainer_state.json').exists():
            return c / 'trainer_state.json'
    return None


def _load(state_path: Path) -> dict:
    state = json.loads(state_path.read_text(encoding='utf-8'))
    hist = state['log_history']
    steps  = [h['step'] for h in hist if 'loss' in h]
    losses = [h['loss'] for h in hist if 'loss' in h]
    return {'steps': steps, 'losses': losses, 'total_flos': state.get('total_flos')}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_state = _final_trainer_state(V2_RUN)
    v3_run = _latest_v3_run()
    v3_state = _final_trainer_state(v3_run) if v3_run else None
    if v2_state is None:
        raise SystemExit(f'no trainer_state under {V2_RUN}')
    if v3_state is None:
        print(f'[plot] no v3 final checkpoint yet (run={v3_run}); skipping v3 overlay')

    v2 = _load(v2_state)
    v3 = _load(v3_state) if v3_state else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(v2['steps'], v2['losses'], color='#fdae6b', linewidth=1.6, label=f'v2 r=32 α=64 (n=1000)')
    if v3:
        ax1.plot(v3['steps'], v3['losses'], color='#a1d99b', linewidth=1.6, label=f'v3 r=64 α=128 (n=2189)')
    ax1.set_xlabel('step'); ax1.set_ylabel('loss')
    ax1.set_title('training loss — v2 vs v3')
    ax1.grid(alpha=0.3); ax1.legend(frameon=False)

    ax2.plot(v2['steps'], v2['losses'], color='#fdae6b', linewidth=1.6, label='v2')
    if v3:
        ax2.plot(v3['steps'], v3['losses'], color='#a1d99b', linewidth=1.6, label='v3')
    ax2.set_yscale('log'); ax2.set_xlabel('step'); ax2.set_ylabel('loss (log)')
    ax2.set_title('same, log-y')
    ax2.grid(alpha=0.3, which='both'); ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'loss_curve_v2_v3.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'loss_curve_v2_v3.svg', bbox_inches='tight')
    plt.close(fig)

    summary = {
        'v2': {
            'run': str(V2_RUN),
            'n_steps': len(v2['steps']),
            'start_loss': v2['losses'][0],
            'end_loss': v2['losses'][-1],
            'min_loss': min(v2['losses']),
            'total_flos': v2['total_flos'],
        },
    }
    if v3:
        summary['v3'] = {
            'run': str(v3_run),
            'n_steps': len(v3['steps']),
            'start_loss': v3['losses'][0],
            'end_loss': v3['losses'][-1],
            'min_loss': min(v3['losses']),
            'total_flos': v3['total_flos'],
        }
    (OUT_DIR / 'loss_summary_v2_v3.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'[plot] wrote loss_curve_v2_v3.{{png,svg}} under {OUT_DIR}')


if __name__ == '__main__':
    main()
