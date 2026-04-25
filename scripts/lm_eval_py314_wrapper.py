"""Python 3.14 + HF datasets dill-compat wrapper for lm-evaluation-harness CLI.

Python 3.14 changed `pickle.Pickler._batch_setitems` to require a third arg
(the full dict), but the `dill` version bundled with HF `datasets` still calls
the 2-arg form. This breaks `datasets.fingerprint.Hasher.hash` which pickles
config dicts to compute cache fingerprints — so every lm_eval run crashes the
moment a task loads its dataset.

Mirrors the same monkey-patch used in scripts/finetune_gemma*_brain.py before
importing HF datasets. See memory/feedback_datasets_py314_dill.md.

Invoke exactly like `python -m lm_eval ...`:
    python scripts/lm_eval_py314_wrapper.py --model hf --model_args ... --tasks ...
"""
from __future__ import annotations

import hashlib
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import datasets.fingerprint as _fp


def _stable_hash(value) -> str:
    return hashlib.sha256(repr(value).encode('utf-8', errors='replace')).hexdigest()


_fp.Hasher.hash = classmethod(lambda cls, value: _stable_hash(value))
_fp.generate_fingerprint = lambda dataset: _stable_hash(id(dataset))

from lm_eval.__main__ import cli_evaluate

if __name__ == '__main__':
    sys.exit(cli_evaluate() or 0)
