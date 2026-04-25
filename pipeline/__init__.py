"""TRIBE v2 / Jemma training + evaluation pipeline.

Priority scheduler with GPU-first ordering, parallel CPU tasks, stall +
timeout detection, self-healing (OOM fallback, HF download retry),
persistent state with resume, and a quick-sanity fast-fail gate.

Entry point: `python -m pipeline <subcommand>`.
"""

__version__ = '0.1.0'
