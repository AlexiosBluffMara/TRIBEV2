"""Download popular Kaggle + HF neuroscience/MRI/brain-related datasets.

Focused on datasets that contain STRUCTURED TEXT (clinical notes, findings,
descriptions), not raw DICOMs. We need text for language-model training.

Each source declares license + one-line rationale. Images are only pulled if
a text description file accompanies them.

Requires Kaggle API creds at C:/Users/soumi/.kaggle/kaggle.json for kaggle
sources. Missing creds means kaggle sources skip silently.

Usage:
    python scripts/pull_kaggle_neuro.py              # pull everything
    python scripts/pull_kaggle_neuro.py --only hf    # HF only, skip kaggle
    python scripts/pull_kaggle_neuro.py --only mriqa_hf,bra_kaggle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import datasets.fingerprint as _fp


def _stable_hash(value) -> str:
    return hashlib.sha256(repr(value).encode('utf-8', errors='replace')).hexdigest()


_fp.Hasher.hash = classmethod(lambda cls, value: _stable_hash(value))
_fp.generate_fingerprint = lambda dataset: _stable_hash(id(dataset))

from datasets import load_dataset


ROOT = Path('D:/research/corpora')
KAGGLE_CRED = Path.home() / '.kaggle' / 'kaggle.json'


class Source:
    def __init__(self, slug: str, backend: str, *, license_id: str,
                 note: str, pull: Callable[[Path], dict], priority: str = 'P2'):
        self.slug = slug
        self.backend = backend  # 'hf' | 'kaggle'
        self.license_id = license_id
        self.note = note
        self.pull = pull
        self.priority = priority


def _pull_hf(repo_id: str, splits: list[str] | None = None,
             config: str | None = None,
             streaming_row_cap: int | None = None) -> Callable[[Path], dict]:
    def _fn(out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        if streaming_row_cap:
            # Stream + cap to a bounded number of rows for huge datasets
            ds = load_dataset(repo_id, config, streaming=True) if config \
                else load_dataset(repo_id, streaming=True)
            counts: dict[str, int] = {}
            for split in (splits or list(ds.keys())):
                rows = []
                for i, row in enumerate(ds[split]):
                    if i >= streaming_row_cap:
                        break
                    rows.append(row)
                counts[split] = len(rows)
                (out_dir / f'{split}.jsonl').write_text(
                    '\n'.join(json.dumps(r, default=str, ensure_ascii=False)
                              for r in rows),
                    encoding='utf-8')
            return {'backend': 'hf', 'repo': repo_id, 'config': config,
                    'streamed': True, 'splits': counts, 'out_dir': str(out_dir)}
        ds = load_dataset(repo_id, config) if config else load_dataset(repo_id)
        counts = {}
        for split in (splits or list(ds.keys())):
            shard = ds[split]
            p = out_dir / f'{split}.parquet'
            shard.to_parquet(str(p))
            counts[split] = len(shard)
        return {'backend': 'hf', 'repo': repo_id, 'config': config,
                'splits': counts, 'out_dir': str(out_dir)}
    return _fn


def _pull_kaggle(dataset_slug: str, unzip: bool = True) -> Callable[[Path], dict]:
    def _fn(out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(out_dir),
                                   unzip=unzip, quiet=False)
        files = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob('*')
                       if p.is_file())
        return {'backend': 'kaggle', 'slug': dataset_slug,
                'files': files[:50], 'out_dir': str(out_dir)}
    return _fn


SOURCES: list[Source] = [
    # === P0: must-have neuroscience / brain narration signal =================
    Source('medmcqa', 'hf',
           license_id='Apache-2.0', priority='P0',
           note='MedMCQA — 194k medical MCQ with explanations across 21 '
                'subjects, neurology well-represented. Commercial-safe.',
           pull=_pull_hf('openlifescienceai/medmcqa')),
    Source('medqa_usmle', 'hf',
           license_id='MIT', priority='P0',
           note='MedQA USMLE 4-options — 12.7k board-style questions with '
                'reasoning. Fully commercial.',
           pull=_pull_hf('GBaker/MedQA-USMLE-4-options')),
    Source('medical_meadow_medqa', 'hf',
           license_id='research', priority='P0',
           note='Medical Meadow MedQA — ~10k instruction-formatted MedQA '
                'variants. Already instruction-shaped: cheap supervision.',
           pull=_pull_hf('medalpaca/medical_meadow_medqa')),
    Source('malikeh_medqa_mix', 'hf',
           license_id='MIT', priority='P0',
           note='Malikeh1375 merged med-QA (MedQA + MedMCQA + PubMedQA + '
                'MedInstruct, 1M+ rows). Single-source shortcut.',
           pull=_pull_hf('Malikeh1375/medical-question-answering-datasets',
                         config='all-processed', streaming_row_cap=50000)),
    Source('cochrane_simplification', 'hf',
           license_id='CC-BY-4.0', priority='P0',
           note='ben-yu/cochrane_combined - ~5k paired technical abstracts '
                'with plain-language Target simplifications. Direct '
                'student/expert bridge. (Mirror of GEM/cochrane-simplification)',
           pull=_pull_hf('ben-yu/cochrane_combined')),
    Source('braingpt_pmc_neuroscience', 'hf',
           license_id='Apache-2.0', priority='P0',
           note='BrainGPT PMC neuroscience 2002-2022 — 332k abstracts + 123k '
                'full-text. 1.3B tokens. Stream first 20k abstracts for now.',
           pull=_pull_hf(
               'BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset',
               streaming_row_cap=20000)),
    # CineBrain deferred — 25GB+ fMRI volumes in 16 files; text stimulus
    # extraction would require domain-specific munging. Pull manually when
    # a targeted use case appears.
    # Source('cinebrain', 'hf',
    #        license_id='Apache-2.0', priority='P0',
    #        note='Fudan-fMRI/CineBrain - audiovisual+fMRI+EEG+ECG on Big Bang '
    #             'Theory. Text stimulus + BOLD fields for narration grounding.',
    #        pull=_pull_hf('Fudan-fMRI/CineBrain')),
    # === P1: strong-fit secondary sources ==================================
    Source('pubmed_oa_commercial', 'hf',
           license_id='Mixed (commercial-safe subset)', priority='P1',
           note='Corran/Pubmed-OpenAccess-Commercial-Use — pre-filtered '
                'commercial-safe PMC papers. Fallback to BrainGPT corpus.',
           pull=_pull_hf('Corran/Pubmed-OpenAccess-Commercial-Use',
                         streaming_row_cap=10000)),
    Source('asset_simplification', 'hf',
           license_id='CC-BY-NC-SA-4.0', priority='P1',
           note='ASSET text simplification — 2.35k Wikipedia sentences with '
                '10 simplifications each. Research-only (NC).',
           pull=_pull_hf('facebook/asset', config='simplification')),
    # scifact (allenai/scifact) is script-based on HF — skipped until mirror.
    # Source('scifact', 'hf',
    #        license_id='CC-BY-NC-2.0', priority='P1',
    #        note='SciFact — 5k scientific claim verification pairs with '
    #             'evidence abstracts. Research-only.',
    #        pull=_pull_hf('allenai/scifact', config='claims')),
    # bc5cdr (bigbio/bc5cdr) is script-based on HF — skipped until mirror.
    # Source('bc5cdr', 'hf',
    #        license_id='CC0-1.0', priority='P1',
    #        note='BioCreative V CDR — 1500 PubMed abstracts with chemical and '
    #             'disease annotations. Public domain.',
    #        pull=_pull_hf('bigbio/bc5cdr')),
    Source('pereira_fmri_passages', 'hf',
           license_id='CC-BY (inferred from source)', priority='P1',
           note='Pereira 2018 passage-level fMRI. Text stimulus paired with '
                'brain response. Very small but directly on-domain.',
           pull=_pull_hf('helena-balabin/pereira_fMRI_passages')),
    Source('pereira_fmri_sentences', 'hf',
           license_id='CC-BY (inferred from source)', priority='P1',
           note='Pereira 2018 sentence-level fMRI. Complements passages.',
           pull=_pull_hf('helena-balabin/pereira_fMRI_sentences')),
    Source('fmri_language_responses', 'hf',
           license_id='unstated (Huth lab)', priority='P1',
           note='imodels/fmri_language_responses — text→voxel-response pairs '
                'in Gallant/Huth family. Highest-signal small dataset.',
           pull=_pull_hf('imodels/fmri_language_responses')),
    # === P2: nice-to-have =============================================
    Source('eeg_semantic_relevance', 'hf',
           license_id='Apache-2.0', priority='P2',
           note='Quoron/EEG-semantic-text-relevance — 23k time-locked '
                'word-level EEG with semantic labels. Per-word salience.',
           pull=_pull_hf('Quoron/EEG-semantic-text-relevance')),
    Source('things_eeg2', 'hf',
           license_id='CC-BY-4.0', priority='P2',
           note='gasparyanartur/things-eeg2 — 10k-100k EEG trials paired '
                'with THINGS image categories. Visual→brain supervision.',
           pull=_pull_hf('gasparyanartur/things-eeg2',
                         streaming_row_cap=10000)),
    Source('simple_english_wikipedia', 'hf',
           license_id='CC-BY-SA-3.0', priority='P2',
           note='Simple English Wikipedia mirror — ~200k articles rewritten '
                'at lower reading level. Student-tier anchor.',
           pull=_pull_hf('Tralalabs/simple-english-wikipedia',
                         streaming_row_cap=20000)),
    Source('openassistant_oasst1', 'hf',
           license_id='Apache-2.0', priority='P2',
           note='OpenAssistant — general instruction tuning. Small slice as '
                'regularizer against catastrophic forgetting.',
           pull=_pull_hf('OpenAssistant/oasst1')),
    # === Kaggle sources (skipped if no creds) =============================
    Source('brain_tumor_mri_kaggle', 'kaggle',
           license_id='CC0-1.0', priority='P2',
           note='masoudnickparvar/brain-tumor-mri-dataset — 7k MRIs, 4 '
                'classes. Images only; pulls as a reference set for multimodal.',
           pull=_pull_kaggle('masoudnickparvar/brain-tumor-mri-dataset')),
    Source('alzheimers_kaggle', 'kaggle',
           license_id='CC-BY-SA-4.0', priority='P2',
           note='Alzheimer\'s disease prediction with clinical notes.',
           pull=_pull_kaggle('brsdincer/alzheimer-features')),
    Source('stroke_kaggle', 'kaggle',
           license_id='CC0-1.0', priority='P2',
           note='Stroke prediction — tabular + risk factor text. Diversity '
                'for medical Q&A.',
           pull=_pull_kaggle('fedesoriano/stroke-prediction-dataset')),
]


def _kaggle_available() -> bool:
    return KAGGLE_CRED.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default='',
                    help='comma-separated slugs (or "hf" / "kaggle" for all of kind)')
    ap.add_argument('--skip', type=str, default='',
                    help='comma-separated slugs or "kaggle"/"hf"')
    ap.add_argument('--priority', type=str, default='',
                    help='only pull these priorities (comma-separated: P0,P1,P2)')
    ap.add_argument('--require-kaggle', action='store_true')
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(',') if s.strip()}
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    priorities = {p.strip() for p in args.priority.split(',') if p.strip()}

    if args.require_kaggle and not _kaggle_available():
        raise SystemExit(
            f'--require-kaggle but {KAGGLE_CRED} not present.\n'
            f'Create API token at https://www.kaggle.com/settings and place '
            f'kaggle.json at that path.')

    manifest: list[dict] = []
    for src in SOURCES:
        if priorities and src.priority not in priorities:
            continue
        if only:
            wanted = (src.slug in only) or (src.backend in only)
            if not wanted:
                continue
        if src.slug in skip or src.backend in skip:
            print(f'[kneuro] skip {src.slug} (in --skip)')
            continue
        if src.backend == 'kaggle' and not _kaggle_available():
            print(f'[kneuro] skip {src.slug} (no kaggle creds; place kaggle.json)')
            continue

        out_dir = ROOT / src.slug
        print(f'\n[kneuro] === {src.slug} ({src.backend}, {src.priority}) ===')
        print(f'[kneuro]     license: {src.license_id}')
        print(f'[kneuro]     note:    {src.note}')
        print(f'[kneuro]     out_dir: {out_dir}')
        try:
            info = src.pull(out_dir)
            info['license'] = src.license_id
            info['note'] = src.note
            info['slug'] = src.slug
            info['priority'] = src.priority
            manifest.append(info)
            (out_dir / 'LICENSE_CARD.md').write_text(
                f'# {src.slug}\n\n'
                f'- **License:** {src.license_id}\n'
                f'- **Backend:** {src.backend}\n'
                f'- **Priority:** {src.priority}\n'
                f'- **Source:** {info.get("repo") or info.get("slug")}\n\n'
                f'{src.note}\n', encoding='utf-8')
            print(f'[kneuro]     OK  {info}')
        except Exception as e:
            print(f'[kneuro]     FAIL {type(e).__name__}: {e}')
            manifest.append({'slug': src.slug, 'priority': src.priority,
                             'error': f'{type(e).__name__}: {e}'})

    manifest_path = ROOT / 'neuro_sources_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\n[kneuro] manifest -> {manifest_path}')

    ok = [m for m in manifest if 'error' not in m]
    bad = [m for m in manifest if 'error' in m]
    print(f'[kneuro] {len(ok)} ok, {len(bad)} failed')
    for m in bad:
        print(f'  - {m["slug"]}: {m["error"]}')


if __name__ == '__main__':
    main()
