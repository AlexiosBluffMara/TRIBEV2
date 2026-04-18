"""Deprecated shim — import from bot.media_gate instead.

Kept only so older call sites don't break. Will be removed in a future release.
"""
from __future__ import annotations

import warnings

from .media_gate import (  # noqa: F401
    DEFAULT,
    MediaDescription,
    MediaDescription as CatClassification,  # back-compat alias
    classify,
)

warnings.warn(
    "bot.cat_gate is deprecated; import bot.media_gate instead.",
    DeprecationWarning,
    stacklevel=2,
)
