"""Measure tier-control: does the model adjust reading level + fact coverage
when asked for student / public / expert answers on the same stimulus?

We hold the user prompt fixed and swap the system prompt across the three
tiers defined in scripts/build_curriculum_v4.py. For each (stimulus, tier) we
compute:
  - Flesch-Kincaid grade level (textstat) of the generated completion
  - Fact overlap vs. a reference expert answer (token set intersection /
    reference token count) as a crude proxy for "did they keep the facts?"
  - Length in words

Result table has one row per (stimulus, model, tier). An adapter "works" if:
  1. FK_student < FK_public < FK_expert (or at least student < expert) by a
     non-trivial margin (>= 1 grade level)
  2. Fact overlap is roughly flat across tiers (if expert hits 0.8 and student
     hits 0.2, we stripped too much; if expert hits 0.8 and student hits 0.75
     the tier head is not actually simplifying)

Stimuli default to a hand-picked 30 from the existing brain v2 eval jsonl plus
20 general-science / medical questions. Pass --stimuli-jsonl to override.

Usage:
    python scripts/eval_tier_control.py \\
        --model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \\
        --peft D:/research/weights/gemma4-31b-brain-curriculum-<ts>/final \\
        --out D:/research/evals/tier_control_<ts>.csv \\
        [--limit 50] [--max-new-tokens 256]

The runner loads the model once and iterates; evaluation is CPU-bound on the
text-stat side and GPU-bound on generation, so this is single-pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import datasets.fingerprint as _fp
import hashlib


def _stable_hash(value) -> str:
    return hashlib.sha256(repr(value).encode('utf-8', errors='replace')).hexdigest()


_fp.Hasher.hash = classmethod(lambda cls, value: _stable_hash(value))
_fp.generate_fingerprint = lambda dataset: _stable_hash(id(dataset))

import torch


TIER_SYSTEM = {
    'student': (
        'You explain things for a curious student (around grade 8). Short sentences, '
        'concrete analogies, no jargon unless you define it in the same breath. Aim '
        'for the reader to feel smart, not lectured at.'),
    'public': (
        'You explain things for an interested adult with no specialist training (a '
        'science journalist or an informed patient). Two or three short paragraphs. '
        'Precise language is fine when it earns its keep. Avoid hedging chains.'),
    'expert': (
        'You explain things for a domain expert (clinician, researcher). Be compact, '
        'use standard terminology, cite structural markers by name. No preamble, no '
        'audience-tier disclaimers.'),
}


def _flesch_kincaid(text: str) -> float:
    """Stand-alone FK grade level: 0.39 * (words/sent) + 11.8 * (syll/word) - 15.59.
    We keep the formula inline to avoid adding textstat just for this."""
    text = text.strip()
    if not text:
        return 0.0
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()] or [text]
    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    words_per_sent = len(words) / len(sentences)
    syll_per_word = syllables / len(words)
    return 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59


_VOWELS = set('aeiouyAEIOUY')


def _count_syllables(word: str) -> int:
    word = word.lower().strip("'-")
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)}


def _fact_overlap(pred: str, ref: str) -> float:
    pt, rt = _tokenize(pred), _tokenize(ref)
    if not rt:
        return 0.0
    return len(pt & rt) / len(rt)


def _load_stimuli(path: Path | None, limit: int) -> list[dict]:
    """Return list of {'stimulus_id', 'prompt', 'reference'} dicts."""
    if path and path.exists():
        rows = [json.loads(l) for l in path.open(encoding='utf-8')]
        rows = rows[:limit]
        out = []
        for i, r in enumerate(rows):
            out.append({
                'stimulus_id': r.get('id') or f's{i:03d}',
                'prompt': r['prompt'],
                'reference': r.get('completion') or r.get('reference') or '',
            })
        return out
    # Fallback: hand-picked generic science + medical questions
    return [
        {'stimulus_id': f's{i:03d}', 'prompt': q, 'reference': ref}
        for i, (q, ref) in enumerate([
            ('Explain why ice floats on water.',
             'Water is one of the few substances that is less dense as a solid than as a liquid. When water freezes, hydrogen bonds arrange molecules into a lattice with more empty space than liquid water, so a given mass of ice occupies more volume and floats.'),
            ('What causes the seasons on Earth?',
             'Earth\'s rotational axis is tilted about 23.5 degrees relative to its orbital plane. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times of year, producing summer and winter.'),
            ('What is the difference between a virus and a bacterium?',
             'Bacteria are independent single-celled organisms with their own metabolism; they can be treated with antibiotics. Viruses are non-cellular genetic packages that require a host cell to replicate; antibiotics do not affect them.'),
            ('How does a nerve cell send a signal?',
             'A neuron generates an action potential: a rapid depolarization caused by sodium influx through voltage-gated channels, followed by repolarization from potassium efflux. The signal propagates along the axon and triggers neurotransmitter release at the synapse.'),
            ('What is inflammation?',
             'Inflammation is the immune system\'s coordinated response to injury or infection, characterized by vasodilation, increased permeability, and recruitment of leukocytes. Cardinal signs are heat, redness, swelling, pain, and loss of function.'),
            ('Why does the Moon always show the same face to Earth?',
             'The Moon is tidally locked: its rotation period equals its orbital period. This synchronous rotation arose from Earth\'s tidal forces dissipating energy from the Moon\'s faster early rotation until rotation matched orbit.'),
            ('What are antibodies?',
             'Antibodies are Y-shaped proteins (immunoglobulins) produced by B cells. Each antibody binds a specific antigen, tagging pathogens for destruction or neutralizing toxins.'),
            ('How does a refrigerator work?',
             'A refrigerator uses a closed loop of refrigerant that absorbs heat from inside the cabinet (evaporating at low pressure) and releases heat outside (condensing at high pressure). A compressor drives the cycle.'),
            ('What is DNA replication?',
             'DNA replication is the process by which a cell copies its genome before dividing. The double helix is unwound by helicase, each strand serves as a template, and DNA polymerase synthesizes complementary strands, producing two identical daughter molecules.'),
            ('What is the function of the mitochondrion?',
             'Mitochondria are organelles that generate most of a cell\'s ATP through oxidative phosphorylation. They also participate in calcium regulation, apoptosis, and have their own small genome.'),
            ('Explain what blood pressure measures.',
             'Blood pressure measures the force exerted by circulating blood on arterial walls, reported as systolic over diastolic in mmHg. Systolic reflects peak pressure during ventricular contraction; diastolic reflects the resting pressure between beats.'),
            ('Why do we see different colors?',
             'Different colors correspond to different wavelengths of visible light. Objects appear colored because pigments absorb some wavelengths and reflect others; the reflected wavelengths reach the eye\'s cone photoreceptors, which signal the brain.'),
            ('What is diabetes?',
             'Diabetes mellitus is a group of disorders marked by elevated blood glucose. Type 1 is autoimmune destruction of insulin-producing beta cells; type 2 involves insulin resistance plus relative insulin deficiency.'),
            ('How do vaccines work?',
             'Vaccines present a harmless form of a pathogen (inactivated, attenuated, subunit, or mRNA) to the immune system, which generates memory B and T cells. On real exposure, the memory response contains the pathogen before illness develops.'),
            ('What is gravity?',
             'Gravity is the mutual attraction between objects with mass. In general relativity, it is the curvature of spacetime caused by mass and energy. For everyday scales, Newton\'s law (F = G m1 m2 / r^2) is accurate.'),
            ('Why is the sky blue?',
             'Rayleigh scattering: air molecules scatter shorter wavelengths of sunlight (blue) more strongly than longer wavelengths. Scattered blue light reaches the observer from many directions, so the sky appears blue.'),
            ('What does the liver do?',
             'The liver filters blood, metabolizes nutrients, detoxifies chemicals, synthesizes plasma proteins like albumin and clotting factors, stores glycogen, and produces bile for fat digestion.'),
            ('How does a wind turbine make electricity?',
             'Wind rotates turbine blades attached to a rotor; the rotor spins a shaft connected to a generator. The generator converts rotational kinetic energy into electrical energy via electromagnetic induction.'),
            ('What is a black hole?',
             'A black hole is a region of spacetime where gravity is so strong that not even light can escape once inside the event horizon. It typically forms from the collapse of a massive star\'s core.'),
            ('What is the immune system?',
             'The immune system is a distributed network of organs, cells, and molecules that detects and eliminates pathogens and abnormal cells. It has an innate arm (fast, general) and an adaptive arm (slow, specific, memory-forming).'),
        ][:limit])
    ]


def _build_prompt(system: str, user: str) -> str:
    """Gemma chat template: <start_of_turn>user ... <end_of_turn><start_of_turn>model"""
    return (f'<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n'
            f'<start_of_turn>model\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True,
                    help='HF model id or local path (base model)')
    ap.add_argument('--peft', type=Path, default=None,
                    help='optional LoRA adapter dir to load on top of base')
    ap.add_argument('--stimuli-jsonl', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--model-tag', type=str, default='',
                    help='optional column value; default derived from --model + --peft')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tag = args.model_tag or (f'{Path(args.model).name}'
                             + (f'+{args.peft.parent.name}' if args.peft else ''))

    print(f'[tier] loading base {args.model}')
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'})

    if args.peft:
        from peft import PeftModel
        print(f'[tier] loading adapter {args.peft}')
        model = PeftModel.from_pretrained(model, str(args.peft))
        model.eval()

    stimuli = _load_stimuli(args.stimuli_jsonl, args.limit)
    print(f'[tier] {len(stimuli)} stimuli × 3 tiers = {len(stimuli) * 3} generations')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stimulus_id', 'tier', 'model_tag',
                    'fk_grade', 'fact_overlap', 'word_count', 'completion'])

        for i, stim in enumerate(stimuli):
            print(f'\n[tier] [{i+1}/{len(stimuli)}] {stim["stimulus_id"]}')
            ref = stim['reference']
            for tier in ('student', 'public', 'expert'):
                sys_msg = TIER_SYSTEM[tier]
                full = _build_prompt(sys_msg, stim['prompt'])
                ids = tok(full, return_tensors='pt').to(model.device)
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        **ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                dt = time.time() - t0
                decoded = tok.decode(out[0][ids['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()
                fk = _flesch_kincaid(decoded)
                ov = _fact_overlap(decoded, ref) if ref else 0.0
                wc = len(re.findall(r"\b[\w'-]+\b", decoded))
                print(f'[tier]     {tier:8s}  FK={fk:5.2f}  overlap={ov:.3f}  '
                      f'wc={wc:3d}  gen={dt:5.1f}s')
                w.writerow([stim['stimulus_id'], tier, tag,
                            f'{fk:.3f}', f'{ov:.4f}', wc, decoded])

    print(f'\n[tier] wrote -> {args.out}')

    # Quick summary
    import collections
    fk_by_tier: dict[str, list[float]] = collections.defaultdict(list)
    ov_by_tier: dict[str, list[float]] = collections.defaultdict(list)
    with args.out.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fk_by_tier[row['tier']].append(float(row['fk_grade']))
            ov_by_tier[row['tier']].append(float(row['fact_overlap']))
    print('\n[tier] Summary (median across stimuli):')
    for t in ('student', 'public', 'expert'):
        vs = sorted(fk_by_tier[t]) or [0.0]
        os_ = sorted(ov_by_tier[t]) or [0.0]
        print(f'  {t:8s}  FK_median={vs[len(vs)//2]:5.2f}  '
              f'overlap_median={os_[len(os_)//2]:.3f}  (n={len(vs)})')


if __name__ == '__main__':
    main()
