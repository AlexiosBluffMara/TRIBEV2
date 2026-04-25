"""Plot v2 training loss curve to D:/TRIBEV2/outputs/paper/figures/.

Reads the final checkpoint's trainer_state.json and emits loss_curve.png + loss_curve.svg.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/plot_v2_training_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


STATE_PATH = Path('D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/checkpoint-375/trainer_state.json')
OUT_DIR    = Path('D:/TRIBEV2/outputs/paper/figures')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    hist = state['log_history']
    steps  = [h['step'] for h in hist if 'loss' in h]
    losses = [h['loss'] for h in hist if 'loss' in h]
    lrs    = [h['learning_rate'] for h in hist if 'loss' in h]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.plot(steps, losses, color='#2b8cbe', linewidth=1.6)
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    ax1.set_title(f'v2 training loss — {steps[0]}->{steps[-1]}: {losses[0]:.3f} to {losses[-1]:.3f}')
    ax1.grid(alpha=0.3)
    # Epoch markers at 125, 250, 375
    for s_bound, label in [(125, 'epoch 1'), (250, 'epoch 2'), (375, 'epoch 3')]:
        ax1.axvline(s_bound, color='#888', linestyle=':', alpha=0.6)

    ax2.plot(steps, losses, color='#e6550d', linewidth=1.6)
    ax2.set_yscale('log')
    ax2.set_xlabel('step')
    ax2.set_ylabel('loss (log)')
    ax2.set_title('same, log-y')
    ax2.grid(alpha=0.3, which='both')

    fig.suptitle(
        'Gemma-3-27B-IT brain-narration QLoRA (r=32, alpha=64, 3 epochs, 1000 rows)',
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'loss_curve.png', dpi=140, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'loss_curve.svg', bbox_inches='tight')
    plt.close(fig)

    # Also dump a compact JSON of key points
    summary = {
        'n_steps': len(steps),
        'start_loss': losses[0],
        'end_loss': losses[-1],
        'min_loss': min(losses),
        'min_loss_step': steps[losses.index(min(losses))],
        'epoch1_end_loss': losses[steps.index(125)] if 125 in steps else None,
        'epoch2_end_loss': losses[steps.index(250)] if 250 in steps else None,
        'epoch3_end_loss': losses[steps.index(375)] if 375 in steps else None,
        'peak_lr': max(lrs),
        'lora_r': 32, 'lora_alpha': 64,
        'epochs': 3, 'dataset_rows': 1000,
        'base_model': 'unsloth/gemma-3-27b-it-bnb-4bit',
        'total_flos': state.get('total_flos'),
    }
    (OUT_DIR / 'loss_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'[plot] wrote {OUT_DIR}/loss_curve.{{png,svg}} and loss_summary.json')


if __name__ == '__main__':
    main()
