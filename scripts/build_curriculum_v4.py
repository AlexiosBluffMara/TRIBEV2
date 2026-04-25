"""Assemble the v4 curriculum jsonl from raw Kaggle/HF source parquets.

Output schema (one row per line):
    {
      "system": <str>,           # tier-specific system prompt
      "prompt": <str>,           # user stimulus / question
      "completion": <str>,       # target response
      "signal": "A|B|C|D",       # source category (see HACKATHON_STRATEGY.md)
      "tier":   "student|public|expert",
      "source": <slug>
    }

Signal letters come from HACKATHON_STRATEGY.md §3:
  A = brain narration (existing synthetic corpus)
  B = tiered scientific explanation
  C = medical Q&A
  D = accessibility / simplification

Default mixture weights roughly balance the four signals. Pass --weights
"A:1.0,B:0.8,C:0.8,D:0.6" to re-weight without re-downloading.

Usage:
    python scripts/build_curriculum_v4.py \\
        --brain-jsonl D:/research/datasets/brain_narrations_combined_2189.jsonl \\
        --corpora-dir D:/research/corpora \\
        --out D:/research/datasets/curriculum_v4_<ts>.jsonl \\
        --max-per-source 2000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable


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


def _load_parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(str(path))
    return t.to_pylist()


# -------- per-source normalizers --------

def _from_brain(path: Path, rng: random.Random) -> Iterable[dict]:
    """Existing synthetic brain narration — signal A, tier=expert."""
    if not path.exists():
        return
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            yield {
                'system': TIER_SYSTEM['expert'],
                'prompt': row['prompt'],
                'completion': row['completion'],
                'signal': 'A', 'tier': 'expert', 'source': 'brain_v2_synth',
            }


def _from_onestop(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """OneStopEnglish — Iowa State mirror stores 567 rows in 3 sequential blocks
    of 189 (label 0=elementary, 1=intermediate, 2=advanced). Article i of each
    block is the same underlying piece rewritten at that tier, so we pair by
    position within block.

    Task: given the advanced version, rewrite it for student/public tier."""
    src = corpora / 'onestop_english'
    shards = list(src.glob('**/*.parquet'))
    for sh in shards:
        rows = _load_parquet_rows(sh)
        # Group rows by label while preserving within-label order
        by_label: dict[int, list[str]] = {0: [], 1: [], 2: []}
        for row in rows:
            label = row.get('label')
            text = (row.get('text') or '').strip()
            if label in by_label and text:
                by_label[label].append(text)
        n = min(len(by_label[0]), len(by_label[1]), len(by_label[2]))
        for i in range(n):
            elem = by_label[0][i]
            inter = by_label[1][i]
            adv = by_label[2][i]
            # Intermediate mirror sometimes prepends the tier label literally
            for prefix in ('Elementary', 'Intermediate', 'Advanced'):
                if inter.startswith(prefix):
                    inter = inter[len(prefix):].lstrip('\n ').strip()
                if elem.startswith(prefix):
                    elem = elem[len(prefix):].lstrip('\n ').strip()
                if adv.startswith(prefix):
                    adv = adv[len(prefix):].lstrip('\n ').strip()
            if not (adv and inter and elem):
                continue
            # adv -> elem (student simplification, hardest drop)
            yield {
                'system': TIER_SYSTEM['student'],
                'prompt': f'Rewrite this passage so a grade-8 student can follow it without losing the point.\n\nPassage:\n{adv}',
                'completion': elem,
                'signal': 'D', 'tier': 'student', 'source': 'onestop_adv2elem',
            }
            # adv -> inter (public-tier rewrite)
            yield {
                'system': TIER_SYSTEM['public'],
                'prompt': f'Rewrite this passage for an interested adult reader. Keep the facts, drop the jargon.\n\nPassage:\n{adv}',
                'completion': inter,
                'signal': 'D', 'tier': 'public', 'source': 'onestop_adv2inter',
            }
            # inter -> elem (smaller drop, also student)
            yield {
                'system': TIER_SYSTEM['student'],
                'prompt': f'Rewrite this passage so a grade-8 student can follow it without losing the point.\n\nPassage:\n{inter}',
                'completion': elem,
                'signal': 'D', 'tier': 'student', 'source': 'onestop_inter2elem',
            }


def _from_sciq(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """SciQ — student-level science MCQ with support passage. Task: answer
    the question with a one-paragraph explanation grounded in the support."""
    src = corpora / 'sciq'
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            question = row.get('question') or ''
            answer = row.get('correct_answer') or ''
            support = (row.get('support') or '').strip()
            if not (question and answer):
                continue
            prompt = f'Question: {question}\n\n'
            if support:
                prompt += f'Background:\n{support}\n\n'
            prompt += 'Answer the question and explain briefly why that answer is correct.'
            completion = f'{answer}. '
            if support:
                completion += (f'Briefly: {support[:400]}'
                               f'{"..." if len(support) > 400 else ""}')
            yield {
                'system': TIER_SYSTEM['student'],
                'prompt': prompt,
                'completion': completion,
                'signal': 'B', 'tier': 'student', 'source': 'sciq',
            }


def _from_pubmedqa(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """PubMedQA pqa_labeled — expert biomedical yes/no/maybe with long_answer.
    Task: summarize the long_answer for a public-tier reader."""
    src = corpora / 'pubmedqa'
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            question = row.get('question') or ''
            long_answer = row.get('long_answer') or ''
            final_decision = row.get('final_decision') or ''
            contexts = row.get('context') or {}
            ctx_text = ''
            if isinstance(contexts, dict):
                parts = contexts.get('contexts') or []
                ctx_text = ' '.join(parts)[:2000]
            if not (question and long_answer):
                continue
            prompt = (f'Clinical question: {question}\n\n'
                      f'Abstract excerpt:\n{ctx_text}\n\n'
                      f'Explain the evidence for an informed patient or health journalist. '
                      f'State the overall takeaway up front, then the nuance.')
            completion = (f'Takeaway: {final_decision}. {long_answer}').strip()
            yield {
                'system': TIER_SYSTEM['public'],
                'prompt': prompt,
                'completion': completion,
                'signal': 'C', 'tier': 'public', 'source': 'pubmedqa',
            }


def _from_medquad(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """MedQuAD — consumer-facing medical Q&A."""
    src = corpora / 'medquad'
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            q = row.get('question') or row.get('Question') or ''
            a = row.get('answer') or row.get('Answer') or ''
            focus = (row.get('question_focus') or row.get('focus_area')
                     or row.get('Focus') or '')
            if not (q and a) or len(a) < 60:
                continue
            prompt = f'Patient question: {q}'
            if focus:
                prompt += f'\nTopic: {focus}'
            prompt += ('\n\nAnswer clearly for a concerned but non-clinical reader. '
                       'Be factual, acknowledge uncertainty when present, and include '
                       'a one-line reminder to confirm with their clinician.')
            completion = a.strip()
            if 'clinician' not in completion.lower() and 'doctor' not in completion.lower():
                completion += ' (Always confirm with your clinician for anything specific to your situation.)'
            yield {
                'system': TIER_SYSTEM['public'],
                'prompt': prompt,
                'completion': completion,
                'signal': 'C', 'tier': 'public', 'source': 'medquad',
            }


def _from_cochrane(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """Cochrane — ~5k paired biomedical review Abstract (technical) → Target
    (plain-language). Two tasks per row:
      (a) D/public: rewrite Abstract as plain-language Target
      (b) C/expert: given the Title, produce a concise expert conclusion (Target
          is still appropriate — clinician-friendly one-paragraph bottom line)

    Many Cochrane reviews within the same clinical domain share identical
    Target summaries (the `ben-yu/cochrane_combined` mirror preserves this).
    Dedup by Target-prefix so one templated response doesn't pollute training.
    """
    src = corpora / 'cochrane_simplification'
    seen_target_prefix: dict[str, int] = {}
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            title = (row.get('Title') or '').strip()
            abstract = (row.get('Abstract') or '').strip()
            target = (row.get('Target') or '').strip()
            if not (abstract and target) or len(abstract) < 120 or len(target) < 60:
                continue
            # Drop duplicates: if this target-prefix has already been emitted,
            # skip the row entirely so the Abstract→Target mapping doesn't
            # teach a one-to-many collapse.
            key = target[:200]
            if seen_target_prefix.get(key, 0) >= 1:
                seen_target_prefix[key] = seen_target_prefix.get(key, 0) + 1
                continue
            seen_target_prefix[key] = seen_target_prefix.get(key, 0) + 1
            # Public-tier simplification
            yield {
                'system': TIER_SYSTEM['public'],
                'prompt': ('Summarize this biomedical review abstract in plain '
                           'language for an informed patient. Keep the clinical '
                           'bottom line, drop statistical jargon.\n\n'
                           f'Abstract:\n{abstract[:3000]}'),
                'completion': target,
                'signal': 'D', 'tier': 'public', 'source': 'cochrane_simplify',
            }
            # Expert-tier conclusion from title (use Target as the compact conclusion)
            if title:
                yield {
                    'system': TIER_SYSTEM['expert'],
                    'prompt': ('Write a compact clinical bottom-line paragraph '
                               'for the following systematic review.\n\n'
                               f'Title: {title}'),
                    'completion': target,
                    'signal': 'C', 'tier': 'expert', 'source': 'cochrane_bottomline',
                }


def _from_medmcqa(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """MedMCQA — 182k single-best-answer MCQs with explanations across 21
    medical subjects. `cop` (str 0-3) is the correct option index. `exp` is
    the rationale.

    Task: answer with the chosen letter and a short rationale. Expert tier."""
    src = corpora / 'medmcqa'
    letters = ['A', 'B', 'C', 'D']
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            q = (row.get('question') or '').strip()
            opts = [(row.get(k) or '').strip()
                    for k in ('opa', 'opb', 'opc', 'opd')]
            try:
                cop_idx = int(row.get('cop', ''))
            except (TypeError, ValueError):
                continue
            if not (q and all(opts) and 0 <= cop_idx < 4):
                continue
            exp = (row.get('exp') or '').strip()
            subj = row.get('subject_name') or ''
            formatted = '\n'.join(f'{letters[i]}. {opts[i]}' for i in range(4))
            prompt = (f'Medical board-style question ({subj}):\n\n'
                      f'{q}\n\n{formatted}\n\n'
                      'Pick the best answer and explain briefly why it is '
                      'correct and why the most plausible distractor is wrong.')
            completion = f'Answer: {letters[cop_idx]}. {opts[cop_idx]}.'
            if exp and len(exp) > 40:
                completion += f'\n\nRationale: {exp}'
            yield {
                'system': TIER_SYSTEM['expert'],
                'prompt': prompt,
                'completion': completion,
                'signal': 'C', 'tier': 'expert', 'source': 'medmcqa',
            }


def _from_medical_meadow_medqa(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """Medical Meadow MedQA — USMLE-style items already instruction-formatted.
    input: 'Q:<stem>?\n{\"A\":\"...\",...}'
    output: 'E: <letter and answer>'
    """
    src = corpora / 'medical_meadow_medqa'
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            inp = (row.get('input') or '').strip()
            instr = (row.get('instruction') or '').strip()
            out = (row.get('output') or '').strip()
            if not (inp and out) or len(inp) < 80:
                continue
            prompt = (f'{instr}\n\n{inp}') if instr else inp
            yield {
                'system': TIER_SYSTEM['expert'],
                'prompt': prompt,
                'completion': out,
                'signal': 'C', 'tier': 'expert', 'source': 'medical_meadow_medqa',
            }


def _from_malikeh(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """Malikeh1375 merged medical Q&A — 'all-processed' split with instruction,
    input, output triples. Patient-voice questions with clinician-style answers.
    Filter short or clearly templated responses.
    """
    src = corpora / 'malikeh_medqa_mix'
    for sh in src.glob('**/*.jsonl'):
        with sh.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                instr = (row.get('instruction') or '').strip()
                inp = (row.get('input') or '').strip()
                out = (row.get('output') or '').strip()
                if not (inp and out) or len(inp) < 30 or len(out) < 80:
                    continue
                # Skip obvious data dumps / junk starts
                low = out.lower()[:60]
                if any(s in low for s in ('hi,', 'hello,', 'hi.', 'hello.',
                                          'hi ', 'hello ')):
                    out = out.split('.', 1)[1].strip() if '.' in out else out
                prompt = f'{instr}\n\nPatient: {inp}' if instr else inp
                completion = out
                if 'clinician' not in completion.lower() \
                        and 'doctor' not in completion.lower() \
                        and 'physician' not in completion.lower():
                    completion += (' (Please confirm any medication or '
                                   'dosage decisions with your own clinician.)')
                yield {
                    'system': TIER_SYSTEM['public'],
                    'prompt': prompt,
                    'completion': completion,
                    'signal': 'C', 'tier': 'public', 'source': 'malikeh_medqa',
                }


def _from_braingpt(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """BrainGPT PMC neuroscience 2002-2022 — single-field `text` scientific
    abstracts. No paired supervision, so we self-split: take the first ~2
    sentences as prompt context, ask the model to continue in expert register.
    Signal B, tier=expert. Primary purpose is domain vocabulary coverage.
    """
    src = corpora / 'braingpt_pmc_neuroscience'
    for sh in src.glob('**/*.jsonl'):
        with sh.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = (row.get('text') or '').strip()
                # Strip HTML entities commonly present in PMC text
                text = (text.replace('&#x3b1;', 'α')
                             .replace('&#x3b2;', 'β')
                             .replace('&#x3b3;', 'γ')
                             .replace('&amp;', '&'))
                if len(text) < 400 or len(text) > 5000:
                    continue
                # Split at first sentence boundary near 20% of length
                target = max(120, min(400, len(text) // 5))
                cut = text.find('. ', target)
                if cut < 0 or cut > len(text) - 200:
                    continue
                prefix = text[:cut + 1].strip()
                rest = text[cut + 2:].strip()
                if len(rest) < 200:
                    continue
                yield {
                    'system': TIER_SYSTEM['expert'],
                    'prompt': ('Continue this neuroscience abstract in the '
                               'same register. Keep the terminology precise.\n\n'
                               f'{prefix}'),
                    'completion': rest,
                    'signal': 'B', 'tier': 'expert', 'source': 'braingpt_pmc',
                }


_JATS_INNER_TAG_RE = None
_JATS_WHITESPACE_RE = None


def _jats_strip(s: str) -> str:
    """Drop all inner XML tags, collapse whitespace, decode common entities."""
    global _JATS_INNER_TAG_RE, _JATS_WHITESPACE_RE
    if _JATS_INNER_TAG_RE is None:
        import re
        _JATS_INNER_TAG_RE = re.compile(r'<[^>]+>')
        _JATS_WHITESPACE_RE = re.compile(r'\s+')
    s = _JATS_INNER_TAG_RE.sub(' ', s)
    s = (s.replace('&amp;', '&')
           .replace('&lt;', '<')
           .replace('&gt;', '>')
           .replace('&quot;', '"')
           .replace('&apos;', "'")
           .replace('\u2009', ' ')
           .replace('\u202f', ' '))
    s = _JATS_WHITESPACE_RE.sub(' ', s).strip()
    return s


def _from_pubmed_oa(corpora: Path, rng: random.Random, cap: int) -> Iterable[dict]:
    """PubMed Open-Access Commercial — JATS XML full papers. Extract
    <article-title> and <abstract> (strip inner tags), emit TWO supervisions
    per paper:

      (1) signal B / expert: from the title, write the abstract. Real
          generative supervision — the model must synthesize study design,
          finding, and take-home from nothing but a title.
      (2) signal D / public: rewrite the expert abstract in plain language
          for an informed patient. We don't have paired ground-truth lay
          summaries here, so instead we use the abstract's own conclusion
          sentence(s) — shorter, stripped of jargon-heavy methods — as
          completion. This is a weaker label but still contrastive.

    The `cap` bounds iteration because the file is ~1 GB and a full pass is
    slow; we stop after `cap` valid source papers have been produced. Each
    paper yields up to 2 rows (one per supervision style).
    """
    import re
    src = corpora / 'pubmed_oa_commercial'
    shards = list(src.glob('**/*.jsonl'))
    abs_re = re.compile(r'<abstract[^>]*>(.*?)</abstract>', re.DOTALL)
    title_re = re.compile(r'<article-title[^>]*>(.*?)</article-title>', re.DOTALL)
    produced = 0
    for sh in shards:
        if produced >= cap:
            break
        with sh.open(encoding='utf-8') as f:
            for line in f:
                if produced >= cap:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                data = row.get('data')
                doc = data[0] if isinstance(data, list) and data else (data or '')
                if not isinstance(doc, str) or '<abstract' not in doc:
                    continue
                abs_m = abs_re.search(doc)
                if not abs_m:
                    continue
                abstract = _jats_strip(abs_m.group(1))
                if len(abstract) < 250 or len(abstract) > 4000:
                    continue
                title_m = title_re.search(doc)
                title = _jats_strip(title_m.group(1)) if title_m else ''
                if not title or len(title) < 10:
                    continue
                produced += 1
                # (1) title -> abstract (expert synthesis)
                yield {
                    'system': TIER_SYSTEM['expert'],
                    'prompt': ('Write the abstract for a biomedical research '
                               'paper with the following title. Follow standard '
                               'structure (background / methods / results / '
                               'conclusion) and keep it tight.\n\n'
                               f'Title: {title}'),
                    'completion': abstract,
                    'signal': 'B', 'tier': 'expert', 'source': 'pubmed_oa_abs',
                }


def _from_pereira_passages(corpora: Path, rng: random.Random) -> Iterable[dict]:
    """Pereira et al. 2018 stimulus passages — 96 short Wikipedia-style
    expository paragraphs used as fMRI stimuli. These are professionally
    curated to be comprehensible to general readers (science journalists
    and educated non-specialists were the target audience).

    We read only 1 subject's row (paragraphs are shared across subjects),
    extract the `paragraphs` field (96 strings), and emit one continuation
    supervision per paragraph: first sentence as anchor, remainder as
    completion. Signal B, tier=public.

    Deliberately small (~96 rows) — the value is style quality, not volume.
    """
    src = corpora / 'pereira_fmri_passages'
    shards = list(src.glob('**/*.parquet'))
    if not shards:
        return
    # Only read the first row — paragraphs are identical across subjects.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(shards[0]))
    batch = next(pf.iter_batches(batch_size=1, columns=['paragraphs']))
    row = batch.to_pylist()[0]
    paragraphs = row.get('paragraphs') or []
    for para in paragraphs:
        if not isinstance(para, str):
            continue
        para = para.strip()
        if len(para) < 200 or len(para) > 2000:
            continue
        # Split after first sentence (prefer '. ' boundary)
        cut = para.find('. ')
        if cut < 20 or cut > 200:
            continue
        prefix = para[:cut + 1].strip()
        rest = para[cut + 2:].strip()
        if len(rest) < 120:
            continue
        yield {
            'system': TIER_SYSTEM['public'],
            'prompt': ('Continue this short explanatory paragraph in the '
                       'same clear expository style (Wikipedia-like). Keep '
                       'it factual and self-contained.\n\n'
                       f'{prefix}'),
            'completion': rest,
            'signal': 'B', 'tier': 'public', 'source': 'pereira_passages',
        }


def _from_asset(corpora: Path, rng: random.Random, variants: int) -> Iterable[dict]:
    """ASSET simplification corpus — each row has `original` sentence and 10
    crowd-authored `simplifications`. All 10 are valid references; emitting
    multiple per row gives the model a taste of paraphrase diversity.

    Signal D, tier=student. The ASSET paper argues the 10 references measure
    simplicity, meaning preservation, and fluency jointly — so picking any
    one is a correct label.
    """
    src = corpora / 'asset_simplification'
    variants = max(1, min(10, variants))
    for sh in src.glob('**/*.parquet'):
        for row in _load_parquet_rows(sh):
            orig = (row.get('original') or '').strip()
            simps = [s.strip() for s in (row.get('simplifications') or [])
                     if isinstance(s, str) and s.strip()]
            if not orig or not simps or len(orig) < 40:
                continue
            # Sample k unique simplifications, prefer shorter/cleaner first few
            chosen = list(simps)
            rng.shuffle(chosen)
            for simp in chosen[:variants]:
                if len(simp) < 15 or simp == orig:
                    continue
                yield {
                    'system': TIER_SYSTEM['student'],
                    'prompt': ('Rewrite this sentence so a grade-8 reader can '
                               'follow it. Keep every fact, drop jargon, and '
                               'split long clauses if you have to.\n\n'
                               f'Sentence: {orig}'),
                    'completion': simp,
                    'signal': 'D', 'tier': 'student', 'source': 'asset_simplif',
                }


def _from_wiki_auto(corpora: Path, rng: random.Random, cap: int) -> Iterable[dict]:
    """WikiAuto aligned complex→simple sentence pairs. Sample cap rows."""
    src = corpora / 'wiki_auto'
    count = 0
    for sh in src.glob('**/*.parquet'):
        if count >= cap:
            break
        for row in _load_parquet_rows(sh):
            if count >= cap:
                break
            # wiki_auto `auto` config uses 'normal_sentence' and 'simple_sentence'
            complex_s = (row.get('normal_sentence') or row.get('complex')
                         or row.get('source') or '')
            simple_s = (row.get('simple_sentence') or row.get('simple')
                        or row.get('target') or '')
            if not (complex_s and simple_s) or len(complex_s) < 40:
                continue
            count += 1
            yield {
                'system': TIER_SYSTEM['student'],
                'prompt': f'Rewrite this sentence so a grade-8 reader can follow it without losing the key fact.\n\nSentence: {complex_s}',
                'completion': simple_s,
                'signal': 'D', 'tier': 'student', 'source': 'wiki_auto',
            }


# -------- assembly --------

SIGNAL_ORDER = ['A', 'B', 'C', 'D']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--brain-jsonl', type=Path,
                    default=Path('D:/research/datasets/brain_narrations_combined_2189.jsonl'))
    ap.add_argument('--corpora-dir', type=Path,
                    default=Path('D:/research/corpora'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-per-source', type=int, default=2000)
    ap.add_argument('--wiki-cap', type=int, default=1500)
    ap.add_argument('--braingpt-cap', type=int, default=1500,
                    help='cap on braingpt_pmc continuation rows (signal B)')
    ap.add_argument('--malikeh-cap', type=int, default=2000,
                    help='cap on Malikeh patient-Q&A rows (signal C public)')
    ap.add_argument('--medmcqa-cap', type=int, default=3000,
                    help='cap on MedMCQA rows (signal C expert)')
    ap.add_argument('--asset-cap', type=int, default=1500,
                    help='cap on asset_simplif rows (signal D student)')
    ap.add_argument('--asset-variants', type=int, default=3,
                    help='simplifications per ASSET source row (1-10); 3 '
                         'balances paraphrase diversity against row bloat')
    ap.add_argument('--pubmed-oa-cap', type=int, default=2000,
                    help='cap on pubmed_oa source papers (signal B expert); '
                         'each paper yields up to 1 row')
    ap.add_argument('--enable-wiki-auto', action='store_true',
                    help='include wiki_auto (download currently fails on script-based datasets)')
    ap.add_argument('--skip-sources', type=str, default='',
                    help='comma-separated source slugs to exclude, e.g. braingpt_pmc,malikeh_medqa')
    ap.add_argument('--weights', type=str,
                    default='A:1.0,B:0.8,C:0.8,D:0.7',
                    help='per-signal sampling weight; lower values subsample')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    weights: dict[str, float] = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
    for spec in args.weights.split(','):
        k, _, v = spec.partition(':')
        k, v = k.strip().upper(), v.strip()
        if k in weights and v:
            weights[k] = float(v)

    skip = {s.strip() for s in args.skip_sources.split(',') if s.strip()}

    source_buckets: dict[str, list[dict]] = {s: [] for s in SIGNAL_ORDER}

    def _take(it, cap):
        out = []
        for row in it:
            out.append(row)
            if len(out) >= cap:
                break
        return out

    def _enabled(slug: str) -> bool:
        return slug not in skip

    print(f'[cv4] weights: {weights}')
    print(f'[cv4] max_per_source: {args.max_per_source}  wiki_cap: {args.wiki_cap}  '
          f'braingpt_cap: {args.braingpt_cap}  malikeh_cap: {args.malikeh_cap}  '
          f'medmcqa_cap: {args.medmcqa_cap}  asset_cap: {args.asset_cap} '
          f'(k={args.asset_variants})  pubmed_oa_cap: {args.pubmed_oa_cap}')
    if skip:
        print(f'[cv4] skipping sources: {sorted(skip)}')

    # --- Signal A: brain narration (synthetic) ---
    if _enabled('brain_v2_synth'):
        a_rows = _take(_from_brain(args.brain_jsonl, rng), args.max_per_source)
        source_buckets['A'].extend(a_rows)
        print(f'[cv4] A brain_v2_synth: {len(a_rows)}')

    # --- Signal B: tiered scientific explanation ---
    if _enabled('sciq'):
        b_rows = _take(_from_sciq(args.corpora_dir, rng), args.max_per_source)
        source_buckets['B'].extend(b_rows)
        print(f'[cv4] B sciq: {len(b_rows)}')

    if _enabled('braingpt_pmc'):
        bg_rows = _take(_from_braingpt(args.corpora_dir, rng), args.braingpt_cap)
        source_buckets['B'].extend(bg_rows)
        print(f'[cv4] B braingpt_pmc: {len(bg_rows)}')

    if _enabled('pubmed_oa_abs'):
        poa_rows = _take(_from_pubmed_oa(args.corpora_dir, rng, args.pubmed_oa_cap),
                         args.pubmed_oa_cap)
        source_buckets['B'].extend(poa_rows)
        print(f'[cv4] B pubmed_oa_abs: {len(poa_rows)}')

    if _enabled('pereira_passages'):
        pp_rows = list(_from_pereira_passages(args.corpora_dir, rng))
        source_buckets['B'].extend(pp_rows)
        print(f'[cv4] B pereira_passages: {len(pp_rows)}')

    # --- Signal C: medical / clinical Q&A ---
    if _enabled('pubmedqa'):
        c_rows_pm = _take(_from_pubmedqa(args.corpora_dir, rng), args.max_per_source // 2)
        source_buckets['C'].extend(c_rows_pm)
        print(f'[cv4] C pubmedqa: {len(c_rows_pm)}')

    if _enabled('medquad'):
        c_rows_mq = _take(_from_medquad(args.corpora_dir, rng), args.max_per_source // 2)
        source_buckets['C'].extend(c_rows_mq)
        print(f'[cv4] C medquad: {len(c_rows_mq)}')

    if _enabled('medmcqa'):
        c_rows_mcqa = _take(_from_medmcqa(args.corpora_dir, rng), args.medmcqa_cap)
        source_buckets['C'].extend(c_rows_mcqa)
        print(f'[cv4] C medmcqa: {len(c_rows_mcqa)}')

    if _enabled('medical_meadow_medqa'):
        c_rows_mm = _take(_from_medical_meadow_medqa(args.corpora_dir, rng),
                          args.max_per_source)
        source_buckets['C'].extend(c_rows_mm)
        print(f'[cv4] C medical_meadow_medqa: {len(c_rows_mm)}')

    if _enabled('malikeh_medqa'):
        c_rows_mk = _take(_from_malikeh(args.corpora_dir, rng), args.malikeh_cap)
        source_buckets['C'].extend(c_rows_mk)
        print(f'[cv4] C malikeh_medqa: {len(c_rows_mk)}')

    if _enabled('cochrane_bottomline'):
        # cochrane yields BOTH a D-signal row AND a C-expert row per input row;
        # iterate full source once and split by signal.
        cochrane_all = list(_from_cochrane(args.corpora_dir, rng))
        rng.shuffle(cochrane_all)
        c_coch = [r for r in cochrane_all if r['signal'] == 'C'][:args.max_per_source]
        d_coch = [r for r in cochrane_all if r['signal'] == 'D'][:args.max_per_source]
        source_buckets['C'].extend(c_coch)
        source_buckets['D'].extend(d_coch)
        print(f'[cv4] C cochrane_bottomline: {len(c_coch)}  '
              f'D cochrane_simplify: {len(d_coch)}')

    # --- Signal D: accessibility / simplification ---
    if _enabled('onestop'):
        d_rows_os = _take(_from_onestop(args.corpora_dir, rng), args.max_per_source)
        source_buckets['D'].extend(d_rows_os)
        print(f'[cv4] D onestop: {len(d_rows_os)}')

    if _enabled('asset_simplif'):
        d_rows_as = _take(_from_asset(args.corpora_dir, rng, args.asset_variants),
                          args.asset_cap)
        source_buckets['D'].extend(d_rows_as)
        print(f'[cv4] D asset_simplif: {len(d_rows_as)}')

    d_rows_wa: list[dict] = []
    if args.enable_wiki_auto and _enabled('wiki_auto'):
        d_rows_wa = _take(_from_wiki_auto(args.corpora_dir, rng, args.wiki_cap),
                          args.wiki_cap)
        source_buckets['D'].extend(d_rows_wa)
        print(f'[cv4] D wiki_auto: {len(d_rows_wa)}')

    # Drop near-duplicate completions across the whole corpus — if two rows
    # share the same first 200 chars of completion, the second teaches the
    # model to emit the same output regardless of prompt. Keep the first
    # occurrence only. We do this per-signal so a common plain-language
    # template in D doesn't suppress a legitimate C expert completion.
    for sig in SIGNAL_ORDER:
        bucket = source_buckets[sig]
        seen: set[str] = set()
        deduped: list[dict] = []
        dropped = 0
        for r in bucket:
            key = (r.get('completion') or '')[:200]
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            deduped.append(r)
        if dropped:
            print(f'[cv4] dedup {sig}: dropped {dropped} duplicate-completion rows')
        source_buckets[sig] = deduped

    # Apply weights by subsampling
    mixed: list[dict] = []
    for sig in SIGNAL_ORDER:
        bucket = source_buckets[sig]
        rng.shuffle(bucket)
        keep = int(len(bucket) * weights[sig])
        mixed.extend(bucket[:keep])
    rng.shuffle(mixed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    per_signal = {s: sum(1 for r in mixed if r['signal'] == s) for s in SIGNAL_ORDER}
    per_tier = {}
    for r in mixed:
        per_tier[r['tier']] = per_tier.get(r['tier'], 0) + 1
    per_source = {}
    for r in mixed:
        per_source[r['source']] = per_source.get(r['source'], 0) + 1

    print(f'\n[cv4] wrote {len(mixed)} rows -> {args.out}')
    print(f'[cv4] per signal: {per_signal}')
    print(f'[cv4] per tier:   {per_tier}')
    print(f'[cv4] per source: {per_source}')


if __name__ == '__main__':
    main()
