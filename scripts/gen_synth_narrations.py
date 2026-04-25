"""Generate synthetic (prompt, narration) pairs via warm Gemma 26B + Schaefer-400 atlas.

Emits D:/research/datasets/brain_narrations_<ts>.jsonl with rows:
    {"prompt": <TRIBE-style prompt>, "completion": <Gemma 26B narration>}

Strictly research-only — output flows into the research finetune track, never into
the commercial bot. See docs/LOCAL_FINETUNE_PLAN.md.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/gen_synth_narrations.py [--n 200]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot import ollama_client
from bot.prompts import PERSONA
from bot.logger import log

# Keep the 26B model warm across all generations — the default keep_alive=0
# in bot.gemma.narrate evicts after every call (30s reload penalty × n rows).
_NARRATION_SYSTEM = (
    f"{PERSONA}\n\n"
    "You are writing a single clinician-facing paragraph. Be concise, "
    "avoid hype, no bullet lists. End with a one-clause reminder that "
    "this is a group-averaged TRIBE v2 prediction, not a diagnostic result."
)

# Plausible, diverse stimulus descriptors across networks.
_STIM_SEEDS = [
    ('a short animated clip of a cat chasing a laser pointer across a wooden floor', 'video'),
    ('a nature documentary segment showing ocean waves crashing on a rocky coastline', 'video'),
    ('a city driving scene at dusk with headlights and neon signs', 'video'),
    ('a lecturer explaining synaptic plasticity with a whiteboard diagram', 'video'),
    ('a silent underwater coral reef with slow-moving fish', 'video'),
    ('an action sequence with rapid cuts and explosions in an urban alley', 'video'),
    ('a monologue in a quiet room delivered to camera', 'video'),
    ('a crowded market with overlapping conversations and ambient clatter', 'audio'),
    ('a piano piece in a concert hall with no visual context', 'audio'),
    ('a read-aloud passage about the history of navigation', 'audio'),
    ('a forest soundscape with birdsong and rustling leaves', 'audio'),
    ('a printed paragraph describing the architecture of the visual cortex', 'text'),
    ('a written narrative about a lone astronaut repairing a satellite', 'text'),
    ('a scientific abstract on hippocampal replay during sleep', 'text'),
    ('a news bulletin transcript about a city marathon', 'text'),
    ('a cooking tutorial filmed in a home kitchen', 'video'),
    ('a sports highlight reel of fast-break basketball plays', 'video'),
    ('a time-lapse of flowers blooming in sequence', 'video'),
    ('a meditation instructor guiding breathing exercises, audio only', 'audio'),
    ('a scene of a dog running through an open meadow toward its owner', 'video'),
    ('a surgical procedure explained step-by-step in third person', 'video'),
    ('a child reading aloud from a picture book', 'audio'),
    ('a chamber ensemble performing a string quartet', 'audio'),
    ('a short silent sequence of abstract colored shapes moving on a plain background', 'video'),
    ('a construction site with cranes, pouring concrete, and rhythmic machine sounds', 'video'),
]

# Biologically plausible network-to-ROI groupings (Schaefer-400, 7Networks).
_NETWORK_ROIS: dict[str, list[str]] = {
    'Vis':            [f'7Networks_{h}_Vis_{i}'       for h in ('LH','RH') for i in range(1, 30)],
    'SomMot':         [f'7Networks_{h}_SomMot_{i}'    for h in ('LH','RH') for i in range(1, 30)],
    'DorsAttn':       [f'7Networks_{h}_DorsAttn_Post_{i}' for h in ('LH','RH') for i in range(1, 12)] +
                      [f'7Networks_{h}_DorsAttn_FEF_{i}'  for h in ('LH','RH') for i in range(1, 6)],
    'SalVentAttn':    [f'7Networks_{h}_SalVentAttn_ParOper_{i}' for h in ('LH','RH') for i in range(1, 6)] +
                      [f'7Networks_{h}_SalVentAttn_TempOccPar_{i}' for h in ('LH','RH') for i in range(1, 4)],
    'Limbic':         [f'7Networks_{h}_Limbic_TempPole_{i}' for h in ('LH','RH') for i in range(1, 5)] +
                      [f'7Networks_{h}_Limbic_OFC_{i}'       for h in ('LH','RH') for i in range(1, 4)],
    'Cont':           [f'7Networks_{h}_Cont_Par_{i}'  for h in ('LH','RH') for i in range(1, 8)] +
                      [f'7Networks_{h}_Cont_pCun_{i}' for h in ('LH','RH') for i in range(1, 4)],
    'Default':        [f'7Networks_{h}_Default_Temp_{i}' for h in ('LH','RH') for i in range(1, 10)] +
                      [f'7Networks_{h}_Default_PFC_{i}'  for h in ('LH','RH') for i in range(1, 10)] +
                      [f'7Networks_{h}_Default_pCunPCC_{i}' for h in ('LH','RH') for i in range(1, 5)],
}

# Modality → network weightings (what kind of stimulus drives what)
_MODALITY_NETS = {
    'video': [('Vis', 0.45), ('DorsAttn', 0.20), ('SomMot', 0.08),
              ('Default', 0.12), ('Cont', 0.08), ('SalVentAttn', 0.05), ('Limbic', 0.02)],
    'audio': [('Default', 0.28), ('SomMot', 0.22), ('Cont', 0.15), ('Vis', 0.05),
              ('DorsAttn', 0.10), ('SalVentAttn', 0.15), ('Limbic', 0.05)],
    'text':  [('Default', 0.40), ('Cont', 0.25), ('DorsAttn', 0.12),
              ('Vis', 0.08), ('SomMot', 0.05), ('SalVentAttn', 0.08), ('Limbic', 0.02)],
}


def _sample_rois(modality: str, rng: random.Random, n: int = 8) -> tuple[list[str], dict[str, float]]:
    mix = _MODALITY_NETS.get(modality, _MODALITY_NETS['video'])
    # Sample ROIs with per-network probability, weighted by mix
    rois: list[str] = []
    tries = 0
    while len(rois) < n and tries < 200:
        net, _ = rng.choices(mix, weights=[w for _, w in mix], k=1)[0]
        candidates = _NETWORK_ROIS.get(net, [])
        if not candidates:
            tries += 1
            continue
        roi = rng.choice(candidates)
        if roi not in rois:
            rois.append(roi)
        tries += 1

    # Synthesize |z| means: larger for first-picked (= higher probability net)
    means: dict[str, float] = {}
    for i, roi in enumerate(rois):
        means[roi] = round(max(0.15, rng.gauss(0.65 - 0.04 * i, 0.10)), 3)
    return rois, means


def _gen_row(seed_idx: int, rng: random.Random) -> dict:
    stim_label, modality = _STIM_SEEDS[seed_idx % len(_STIM_SEEDS)]
    top_rois, roi_means = _sample_rois(modality, rng, n=8)
    duration_s   = round(rng.uniform(12.0, 75.0), 1)
    peak_time_s  = round(rng.uniform(2.0, duration_s - 2.0), 1)

    roi_lines = "\n".join(f"  - {r}: mean |z| = {roi_means[r]:.3f}" for r in top_rois)
    prompt = (
        f"Stimulus: {stim_label}\n"
        f"Duration: {duration_s:.1f} s, peak activity at t={peak_time_s:.1f}s.\n"
        f"Top Schaefer-400 regions by mean |z|:\n{roi_lines}\n\n"
        "In 3-5 sentences, explain what this activation pattern suggests the "
        "brain is doing, grouping regions into networks where possible."
    )

    completion = ollama_client.generate(
        prompt      = prompt,
        system      = _NARRATION_SYSTEM,
        model       = 'gemma4:26b',
        num_predict = 400,
        temperature = 0.45,
        keep_alive  = '15m',
    )

    return {
        'stimulus_label': stim_label,
        'modality':       modality,
        'top_rois':       top_rois,
        'roi_means':      roi_means,
        'duration_s':     duration_s,
        'peak_time_s':    peak_time_s,
        'prompt':         prompt,
        'completion':     completion.strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    out_dir = Path('D:/research/datasets')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'brain_narrations_{int(time.time())}.jsonl'

    rng = random.Random(args.seed)
    print(f'[synth] target n = {args.n}; out = {out_path}')

    t0 = time.time()
    with out_path.open('w', encoding='utf-8') as f:
        for i in range(args.n):
            ts = time.time()
            try:
                row = _gen_row(i, rng)
            except Exception as exc:
                log.warning('[synth] row %d failed: %s', i, exc)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            dt = time.time() - ts
            if (i + 1) % 10 == 0 or i == 0:
                print(f'[synth] {i+1}/{args.n}  ({dt:.1f}s/row, total {time.time()-t0:.0f}s)')

    total = time.time() - t0
    print(f'[synth] DONE: {args.n} rows in {total/60:.1f} min -> {out_path}')


if __name__ == '__main__':
    main()
