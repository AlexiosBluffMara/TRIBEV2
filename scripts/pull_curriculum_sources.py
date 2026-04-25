"""Download + normalize curriculum training sources into raw per-dataset dirs.

Supports two fetch backends: Hugging Face Hub datasets and Kaggle. Each source
has a declared license + one-line rationale so the `docs/datasets/` license
cards can be generated after download.

We pull the raw data only — normalization into the curriculum jsonl is done
by scripts/build_curriculum_v4.py so that mixture weights / filters can be
re-tweaked without re-downloading.

Usage:
    python scripts/pull_curriculum_sources.py              # pull everything
    python scripts/pull_curriculum_sources.py --only onestop_english,sciq
    python scripts/pull_curriculum_sources.py --skip kaggle

Kaggle requires `C:/Users/soumi/.kaggle/kaggle.json` credentials; if missing,
kaggle datasets are silently skipped unless --require-kaggle is set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault('HF_HOME', 'C:/Users/soumi/.cache/huggingface')

import datasets.fingerprint as _fp
import hashlib


def _stable_hash(value) -> str:
    return hashlib.sha256(repr(value).encode('utf-8', errors='replace')).hexdigest()


_fp.Hasher.hash = classmethod(lambda cls, value: _stable_hash(value))
_fp.generate_fingerprint = lambda dataset: _stable_hash(id(dataset))

from datasets import load_dataset


ROOT = Path('D:/research/corpora')
KAGGLE_CRED = Path.home() / '.kaggle' / 'kaggle.json'


class Source:
    def __init__(self, slug: str, backend: str, *, license_id: str,
                 note: str, pull: Callable[[Path], dict]):
        self.slug = slug
        self.backend = backend  # 'hf' | 'kaggle'
        self.license_id = license_id
        self.note = note
        self.pull = pull


def _pull_hf(repo_id: str, splits: list[str] | None = None,
             config: str | None = None) -> Callable[[Path], dict]:
    def _fn(out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(repo_id, config) if config else load_dataset(repo_id)
        counts: dict[str, int] = {}
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
        # Lazy-import kaggle so the whole script works without creds if
        # the user only wants HF sources.
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
    # Signal D — direct three-tier readability match (our star dataset)
    Source('onestop_english', 'hf',
           license_id='CC-BY-SA-4.0',
           note='Iowa State mirror of OneStopEnglish — elementary/intermediate/advanced '
                'tiers of the same article. Direct match for student/public/expert tiers.',
           pull=_pull_hf('iastate/onestop_english')),
    # Signal B — student science
    Source('sciq', 'hf',
           license_id='CC-BY-NC-3.0',
           note='AllenAI SciQ — 13.7k crowdsourced science MCQ with explanation '
                'distractor hints. Research-only (CC-NC), keep segregated.',
           pull=_pull_hf('allenai/sciq')),
    # Signal C — medical reasoning (MIT, commercial-safe)
    Source('pubmedqa', 'hf',
           license_id='MIT',
           note='PubMedQA expert-labeled (1k) + artificial (211k) biomedical '
                'Q&A over abstracts. Use pqa_labeled only for training to stay '
                'high quality.',
           pull=_pull_hf('qiaojin/PubMedQA', config='pqa_labeled')),
    # Signal C — consumer medical Q&A
    Source('medquad', 'hf',
           license_id='CC-BY-SA-4.0',
           note='Lavita mirror of NIH/NCI MedQuAD — 16 sources, consumer-facing '
                'questions with structured answers.',
           pull=_pull_hf('lavita/MedQuAD')),
    # Signal D — sentence-level simplification (for style transfer learning)
    Source('wiki_auto', 'hf',
           license_id='CC-BY-SA-3.0',
           note='WikiAuto — 488k automatically aligned complex/simple sentence '
                'pairs. Use a 10% sample; attribution + share-alike required.',
           pull=_pull_hf('wiki_auto', config='auto')),
]


def _kaggle_available() -> bool:
    return KAGGLE_CRED.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default='',
                    help='comma-separated slugs; default is all')
    ap.add_argument('--skip', type=str, default='',
                    help='comma-separated slugs or "kaggle" to skip all Kaggle')
    ap.add_argument('--require-kaggle', action='store_true')
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(',') if s.strip()}
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}

    if args.require_kaggle and not _kaggle_available():
        raise SystemExit(
            f'--require-kaggle but {KAGGLE_CRED} not present.\n'
            f'Create API token at https://www.kaggle.com/settings and place the '
            f'downloaded kaggle.json at that path.')

    manifest: list[dict] = []
    for src in SOURCES:
        if only and src.slug not in only:
            continue
        if src.slug in skip:
            print(f'[pull] skip {src.slug} (in --skip)')
            continue
        if src.backend == 'kaggle' and ('kaggle' in skip or not _kaggle_available()):
            print(f'[pull] skip {src.slug} (no kaggle creds; place kaggle.json)')
            continue

        out_dir = ROOT / src.slug
        print(f'\n[pull] === {src.slug} ({src.backend}) ===')
        print(f'[pull]     license: {src.license_id}')
        print(f'[pull]     note:    {src.note}')
        print(f'[pull]     out_dir: {out_dir}')
        try:
            info = src.pull(out_dir)
            info['license'] = src.license_id
            info['note'] = src.note
            info['slug'] = src.slug
            manifest.append(info)
            (out_dir / 'LICENSE_CARD.md').write_text(
                f'# {src.slug}\n\n'
                f'- **License:** {src.license_id}\n'
                f'- **Backend:** {src.backend}\n'
                f'- **Source:** {info.get("repo") or info.get("slug")}\n\n'
                f'{src.note}\n', encoding='utf-8')
            print(f'[pull]     OK  {info}')
        except Exception as e:
            print(f'[pull]     FAIL {type(e).__name__}: {e}')
            manifest.append({'slug': src.slug, 'error': f'{type(e).__name__}: {e}'})

    manifest_path = ROOT / 'curriculum_sources_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\n[pull] manifest -> {manifest_path}')

    bad = [m for m in manifest if 'error' in m]
    if bad:
        print(f'[pull] {len(bad)}/{len(manifest)} sources failed:')
        for m in bad:
            print(f'  - {m["slug"]}: {m["error"]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
