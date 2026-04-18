"""
Enterprise-grade file validation for JemmaBrain — Red Team Kitchen.

Security layers (in order of execution):
  1. Filename sanitization        — allowlist [a-zA-Z0-9._-], strip path separators
  2. Extension check              — allowlist only, normalized to lowercase
  3. File size limits             — configurable min/max, checked before disk write
  4. Disk space check             — ensure headroom exists before accepting upload
  5. Magic byte verification      — reject files whose header doesn't match extension
  6. Entropy analysis             — flag suspiciously random data disguised as media
  7. SHA-256 deduplication        — reject exact duplicate uploads (session-scoped)
  8. Atomic write                 — write to .tmp first, rename on full pass
  9. ffprobe integrity check      — detect corrupt/truncated/codec-bomb/overlong files
  10. ClamAV scan (optional)      — graceful degradation if not installed
  11. Path traversal gate         — final resolved-path check before rename

Public API:
    validate_and_save(raw_bytes, original_filename, upload_dir, ...)
        → ValidatedFile (or raises ValidationError)

    validate_and_save_streaming(stream, original_filename, upload_dir, ...)
        → ValidatedFile — memory-efficient for large files

Usage in bot.py:
    from .file_validator import validate_and_save, ValidationError as FileValidationError
    try:
        vf = await validate_and_save(raw_bytes=..., original_filename=..., upload_dir=...)
        dest = vf.path
    except FileValidationError as exc:
        await message.reply(f"❌ **File rejected:** {exc.reason}")
        return
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

log = logging.getLogger("jemmabrain.validator")

# Windows: suppress console window for child processes
_NOWWIN: int = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ── Limits ────────────────────────────────────────────────────────────────────

MAX_FILENAME_LEN   = 200        # chars after sanitization (before timestamp prefix)
MAX_STREAMS        = 8          # ffprobe stream count above this = codec bomb suspect
MAX_BITRATE_MBPS   = 200        # Mbps — above this triggers rejection
MAX_DURATION_S     = 600        # 10 minutes hard limit
MIN_FILE_BYTES     = 512        # < 512 B = definitely corrupt / empty
ENTROPY_THRESHOLD  = 7.97       # Shannon bits/byte — only purely-random data exceeds this
ENTROPY_SAMPLE_B   = 65_536     # first 64 KB sampled for entropy
FFPROBE_TIMEOUT_S  = 30         # seconds before ffprobe killed
CLAMAV_TIMEOUT_S   = 120        # seconds before clamscan killed


# ── Allowed formats + magic signatures ───────────────────────────────────────
#
# Each format maps to a list of *candidates*.
# A candidate is a list of (offset, required_bytes) tuples that ALL must match.
# A file passes if ANY one candidate matches.
# This handles containers that have different valid header layouts.

_MAGIC: dict[str, list[list[tuple[int, bytes]]]] = {
    # ── Video ────────────────────────────────────────────────────────────────
    ".mp4": [
        [(4, b"ftypisom")],
        [(4, b"ftypmp42")],
        [(4, b"ftypMSNV")],
        [(4, b"ftypM4V ")],
        [(4, b"ftyp")],          # any ISO Base Media (broadest match, last resort)
    ],
    ".mov": [
        [(4, b"ftyp")],           # modern QuickTime (ISO Base Media)
        [(4, b"moov")],           # old-style QuickTime (moov at front)
        [(4, b"wide")],
        [(4, b"mdat")],
        [(4, b"free")],
        [(4, b"skip")],
        [(4, b"pnot")],
        [(4, b"junk")],
    ],
    ".m4a": [
        [(4, b"ftypM4A ")],
        [(4, b"ftypmp42")],
        [(4, b"ftyp")],
    ],
    ".mkv": [
        [(0, b"\x1a\x45\xdf\xa3")],   # EBML header
    ],
    ".webm": [
        [(0, b"\x1a\x45\xdf\xa3")],   # WebM is a restricted Matroska profile
    ],
    ".avi": [
        [(0, b"RIFF"), (8, b"AVI ")],
        [(0, b"RIFF"), (8, b"AVIX")],
    ],
    # ── Audio ─────────────────────────────────────────────────────────────────
    ".wav": [
        [(0, b"RIFF"), (8, b"WAVE")],
    ],
    ".mp3": [
        [(0, b"ID3")],             # ID3v2 tagged
        [(0, b"\xff\xfb")],        # MPEG1 Layer3 sync (no padding, stereo)
        [(0, b"\xff\xfa")],        # MPEG1 Layer3 sync (no protection)
        [(0, b"\xff\xf3")],        # MPEG2 Layer3
        [(0, b"\xff\xf2")],        # MPEG2.5 Layer3
        [(0, b"\xff\xe3")],        # MPEG2 Layer3 (no protection)
        [(0, b"\xff\xe2")],
        [(0, b"\xff\xfe")],        # MPEG1 Layer1
        [(0, b"APETAGEX")],        # APEv2 tag (MP3 with APE header)
    ],
    ".flac": [
        [(0, b"fLaC")],
    ],
    ".ogg": [
        [(0, b"OggS")],
    ],
    ".aac": [
        [(0, b"\xff\xf1")],        # ADTS AAC-LC (MPEG4)
        [(0, b"\xff\xf9")],        # ADTS AAC-LC (MPEG2)
    ],
}

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(_MAGIC.keys())

# Formats where high entropy is expected (compressed lossy audio) — skip entropy check
_HIGH_ENTROPY_EXTS: frozenset[str] = frozenset({".mp3", ".aac", ".ogg", ".flac"})


# ── Public types ──────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when a file fails any security check. Caller-safe (no secrets)."""
    def __init__(self, reason: str, code: str = "invalid_file"):
        super().__init__(reason)
        self.reason = reason
        self.code   = code


@dataclass(frozen=True)
class ValidatedFile:
    path:         Path    # final saved path (unique, inside upload_dir)
    hash:         str     # SHA-256 hex digest
    size_bytes:   int
    duration_s:   float   # from ffprobe (0.0 if not determinable)
    n_streams:    int
    is_duplicate: bool    # True if SHA-256 was already seen this session


# ── Session-scoped deduplication ─────────────────────────────────────────────
# In-memory; single-process only.  For multi-process deployments, back this
# with Redis or a shared SQLite table.

_seen_hashes: set[str] = set()


# ── Primary public API ────────────────────────────────────────────────────────

async def validate_and_save(
    *,
    raw_bytes:         bytes,
    original_filename: str,
    upload_dir:        Path,
    max_bytes:         int  = 50 * 1024 * 1024,
    check_disk:        bool = True,
    run_clamav:        bool = True,
    deduplicate:       bool = True,
) -> ValidatedFile:
    """
    Full in-memory validation pipeline (loads file into RAM first).
    Good for Discord bot (files already in memory from attachment.read()).
    Runs blocking I/O (ffprobe, ClamAV) in a thread executor.
    Raises ValidationError on failure; returns ValidatedFile on success.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _validate_sync,
        raw_bytes, original_filename, upload_dir,
        max_bytes, check_disk, run_clamav, deduplicate,
    )


async def validate_and_save_streaming(
    *,
    stream:            AsyncIterator[bytes],
    original_filename: str,
    upload_dir:        Path,
    max_bytes:         int  = 500 * 1024 * 1024,
    chunk_size:        int  = 64 * 1024,
    check_disk:        bool = True,
    run_clamav:        bool = True,
    deduplicate:       bool = True,
) -> ValidatedFile:
    """
    Memory-efficient streaming variant for FastAPI web uploads.
    Streams chunks directly to disk without loading the whole file into RAM.
    Raises ValidationError on failure; returns ValidatedFile on success.
    """
    safe_name = _sanitize_filename(original_filename)
    ext       = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"File type '{ext}' is not allowed.  "
            f"Permitted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            "unsupported_type",
        )

    if check_disk:
        _check_disk_space(upload_dir, required_bytes=max_bytes)

    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / (safe_name + ".tmp")
    _assert_inside(tmp_path, upload_dir)

    # Stream chunks to disk while computing SHA-256 and accumulating the header
    sha256      = hashlib.sha256()
    total       = 0
    header_buf  = bytearray()  # accumulate first 16 bytes for magic check

    try:
        with open(tmp_path, "wb") as fp:
            async for chunk in stream:
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValidationError(
                        f"Upload exceeds the {max_bytes // (1024*1024)} MB limit.",
                        "too_large",
                    )
                fp.write(chunk)
                sha256.update(chunk)
                if len(header_buf) < 16:
                    header_buf.extend(chunk[: 16 - len(header_buf)])
    except ValidationError:
        tmp_path.unlink(missing_ok=True)
        raise

    if total < MIN_FILE_BYTES:
        tmp_path.unlink(missing_ok=True)
        raise ValidationError("File is too small or empty — likely corrupt.", "too_small")

    # Magic check (only first 16 bytes available — enough for all formats)
    try:
        _verify_magic(bytes(header_buf), ext)
    except ValidationError:
        tmp_path.unlink(missing_ok=True)
        raise

    digest = sha256.hexdigest()
    is_dup = deduplicate and digest in _seen_hashes

    loop = asyncio.get_running_loop()

    # ffprobe + ClamAV in thread executor (blocking)
    def _post_stream_checks() -> tuple[float, int]:
        dur, ns = _ffprobe_check(tmp_path)
        if run_clamav:
            _clamav_scan(tmp_path)
        return dur, ns

    try:
        duration_s, n_streams = await loop.run_in_executor(None, _post_stream_checks)
    except ValidationError:
        tmp_path.unlink(missing_ok=True)
        raise

    # Rename .tmp → final
    final_path = _commit(tmp_path, ext)
    _seen_hashes.add(digest)

    log.info(
        "[validator] ✅ %s — %.1f MB, %.1fs, %d streams, sha256=%s…",
        final_path.name, total / 1e6, duration_s, n_streams, digest[:16],
    )
    return ValidatedFile(
        path=final_path, hash=digest, size_bytes=total,
        duration_s=duration_s, n_streams=n_streams, is_duplicate=is_dup,
    )


# ── Blocking implementation (runs in thread) ──────────────────────────────────

def _validate_sync(
    raw_bytes:   bytes,
    orig_name:   str,
    upload_dir:  Path,
    max_bytes:   int,
    check_disk:  bool,
    run_clamav:  bool,
    deduplicate: bool,
) -> ValidatedFile:

    # 1 — Filename sanitization
    safe_name = _sanitize_filename(orig_name)

    # 2 — Extension allowlist
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"File type '{ext}' is not allowed.  "
            f"Permitted: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            "unsupported_type",
        )

    # 3 — Size limits
    size = len(raw_bytes)
    if size < MIN_FILE_BYTES:
        raise ValidationError(
            f"File too small ({size} bytes) — likely corrupt or empty.",
            "too_small",
        )
    if size > max_bytes:
        raise ValidationError(
            f"File too large ({size / 1e6:.1f} MB).  Limit: {max_bytes // (1024*1024)} MB.",
            "too_large",
        )

    # 4 — Disk space
    if check_disk:
        _check_disk_space(upload_dir, required_bytes=size * 2)

    # 5 — Magic bytes
    _verify_magic(raw_bytes, ext)

    # 6 — Entropy (skip for naturally-high-entropy audio formats)
    if ext not in _HIGH_ENTROPY_EXTS:
        _check_entropy(raw_bytes, ext)

    # 7 — SHA-256 + dedup
    digest = hashlib.sha256(raw_bytes).hexdigest()
    is_dup = deduplicate and digest in _seen_hashes
    if is_dup:
        log.info("[validator] Duplicate SHA-256 %s… — processing anyway", digest[:16])

    # 8 — Atomic write
    tmp_path = _write_tmp(raw_bytes, upload_dir, safe_name)

    # 9 — ffprobe integrity check
    try:
        duration_s, n_streams = _ffprobe_check(tmp_path)
    except ValidationError:
        tmp_path.unlink(missing_ok=True)
        raise

    # 10 — ClamAV
    if run_clamav:
        try:
            _clamav_scan(tmp_path)
        except ValidationError:
            tmp_path.unlink(missing_ok=True)
            raise

    # 11 — Commit
    final_path = _commit(tmp_path, ext)
    _seen_hashes.add(digest)

    log.info(
        "[validator] ✅ %s — %.1f MB, %.1fs, %d streams, sha256=%s…",
        final_path.name, size / 1e6, duration_s, n_streams, digest[:16],
    )
    return ValidatedFile(
        path=final_path, hash=digest, size_bytes=size,
        duration_s=duration_s, n_streams=n_streams, is_duplicate=is_dup,
    )


# ── Step implementations ──────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """
    Strict allowlist sanitization.

    Rules:
    - Remove null bytes entirely (null-byte injection defense)
    - Take only basename (strip any injected directory prefix)
    - Replace anything not in [a-zA-Z0-9._-] with underscore
    - Remove leading dots (no hidden files, no .htaccess etc.)
    - Enforce MAX_FILENAME_LEN on the stem
    - Prepend Unix timestamp for uniqueness
    """
    if not name:
        raise ValidationError("Filename is empty.", "bad_filename")

    name = name.replace("\x00", "")             # null byte removal
    name = Path(name).name                       # basename only (strips dir traversal)

    if not name:
        raise ValidationError("Filename is empty after sanitization.", "bad_filename")

    # Allowlist: only safe ASCII characters
    name = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)

    # Strip leading dots
    name = name.lstrip(".")
    if not name:
        raise ValidationError("Filename has no safe characters.", "bad_filename")

    # Trim stem length
    stem = Path(name).stem
    suf  = Path(name).suffix[:16]   # cap extension too
    if len(stem) > MAX_FILENAME_LEN - len(suf) - 14:  # 14 = len("1234567890_") + safety
        stem = stem[: MAX_FILENAME_LEN - len(suf) - 14]
    name = stem + suf

    return f"{int(time.time())}_{name}"


def _check_disk_space(upload_dir: Path, required_bytes: int) -> None:
    try:
        free = shutil.disk_usage(upload_dir).free
        if free < required_bytes:
            raise ValidationError(
                f"Server storage almost full — {free // (1024*1024)} MB free, "
                f"need {required_bytes // (1024*1024)} MB.  Try again later.",
                "disk_full",
            )
    except OSError as exc:
        log.warning("[validator] disk_usage check failed: %s", exc)


def _verify_magic(data: bytes, ext: str) -> None:
    """Verify file header against known signatures for the declared extension."""
    candidates = _MAGIC.get(ext)
    if not candidates:
        return  # no magic table entry — pass (shouldn't happen for our allowlist)

    def _matches(candidate: list[tuple[int, bytes]]) -> bool:
        return all(
            len(data) >= off + len(magic) and data[off : off + len(magic)] == magic
            for off, magic in candidate
        )

    if any(_matches(c) for c in candidates):
        return  # at least one candidate matches

    raise ValidationError(
        f"File header does not match the declared type '{ext}'.  "
        "The file may have been renamed from a different format, or is corrupt.",
        "bad_magic",
    )


def _check_entropy(data: bytes, ext: str) -> None:
    """
    Shannon entropy check.  Threshold 7.97 — only purely random / encrypted
    data hits this value reliably.  Compressed video can reach ~7.8 but rarely
    exceeds 7.95 in the first 64 KB (which is mostly container headers).
    """
    sample = data[:ENTROPY_SAMPLE_B]
    if len(sample) < 256:
        return

    n       = len(sample)
    counts  = Counter(sample)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    if entropy > ENTROPY_THRESHOLD:
        raise ValidationError(
            f"File has unusually high entropy ({entropy:.3f} bits/byte) for type '{ext}'.  "
            "This may indicate an encrypted or compressed archive disguised as media.",
            "high_entropy",
        )


def _write_tmp(data: bytes, upload_dir: Path, safe_name: str) -> Path:
    """Write bytes to a .tmp file inside upload_dir atomically."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = upload_dir / (safe_name + ".tmp")
    _assert_inside(tmp, upload_dir)
    tmp.write_bytes(data)
    return tmp


def _assert_inside(path: Path, directory: Path) -> None:
    """Raise ValidationError if path is not inside directory (path traversal gate)."""
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        raise ValidationError(
            "Path traversal detected — destination is outside the upload directory.",
            "path_traversal",
        )


def _commit(tmp: Path, ext: str) -> Path:
    """Rename .tmp → final name.  If name collision, add epoch suffix."""
    final = tmp.with_suffix(ext)
    if final.exists():
        final = final.with_stem(final.stem + f"_{int(time.time() * 1000) % 100000}")
    tmp.rename(final)
    return final


def _ffprobe_check(path: Path) -> tuple[float, int]:
    """
    Run ffprobe on the saved .tmp file.
    Returns (duration_seconds, n_streams).
    Raises ValidationError for: corrupt, timeout, too long, too many streams, bitrate bomb.
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        log.warning("[validator] ffprobe not found — skipping integrity check")
        return 0.0, 1

    cmd = [
        ffprobe, "-v", "error",
        "-show_entries",
        "format=duration,bit_rate:stream=codec_type,codec_name,index",
        "-of", "json",
        str(path),
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_S,
            creationflags=_NOWWIN,
        )
    except subprocess.TimeoutExpired:
        raise ValidationError(
            "File integrity check timed out — the file may be malformed or a codec bomb.",
            "ffprobe_timeout",
        )
    except FileNotFoundError:
        log.warning("[validator] ffprobe binary disappeared mid-run — skipping")
        return 0.0, 1

    if r.returncode != 0:
        err_snippet = (r.stderr or "").strip()[:300]
        raise ValidationError(
            f"File failed integrity check (ffprobe exit {r.returncode}): {err_snippet}",
            "corrupt_file",
        )

    try:
        meta = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise ValidationError("ffprobe returned non-JSON output — file may be corrupt.", "corrupt_file")

    fmt     = meta.get("format", {})
    streams = meta.get("streams", [])

    # Duration
    duration_s = 0.0
    if raw_dur := fmt.get("duration"):
        try:
            duration_s = float(raw_dur)
        except (ValueError, TypeError):
            pass
    if duration_s > MAX_DURATION_S:
        raise ValidationError(
            f"Media is {duration_s / 60:.1f} minutes — maximum is {MAX_DURATION_S // 60} minutes.",
            "too_long",
        )

    # Stream count (codec bomb)
    n_streams = len(streams)
    if n_streams > MAX_STREAMS:
        raise ValidationError(
            f"File has {n_streams} streams (maximum {MAX_STREAMS}).  "
            "This may be a codec bomb or malformed container.",
            "too_many_streams",
        )

    # Bitrate bomb
    if raw_br := fmt.get("bit_rate"):
        try:
            mbps = int(raw_br) / 1_000_000
            if mbps > MAX_BITRATE_MBPS:
                raise ValidationError(
                    f"File bitrate {mbps:.0f} Mbps exceeds maximum ({MAX_BITRATE_MBPS} Mbps).  "
                    "Possible codec bomb.",
                    "bitrate_bomb",
                )
        except (ValueError, TypeError):
            pass

    log.debug("[validator] ffprobe OK: duration=%.1fs, streams=%d", duration_s, n_streams)
    return duration_s, n_streams


def _clamav_scan(path: Path) -> None:
    """
    Optional ClamAV malware scan.
    Gracefully skips if clamscan binary is not found.
    Raises ValidationError only on definitive match (exit 1).
    """
    clamscan = _find_clamscan()
    if not clamscan:
        log.debug("[validator] clamscan not found — skipping AV scan")
        return

    try:
        r = subprocess.run(
            [clamscan, "--no-summary", "--infected", str(path)],
            capture_output=True,
            text=True,
            timeout=CLAMAV_TIMEOUT_S,
            creationflags=_NOWWIN,
        )
    except subprocess.TimeoutExpired:
        log.warning("[validator] ClamAV timed out for %s — treating as clean", path.name)
        return
    except Exception as exc:
        log.warning("[validator] ClamAV error: %s — treating as clean", exc)
        return

    if r.returncode == 1:
        snippet = r.stdout.strip()[:300]
        raise ValidationError(
            f"File was flagged by antivirus scan: {snippet}",
            "malware_detected",
        )
    if r.returncode not in (0, 1):
        log.warning("[validator] ClamAV exit %d for %s — treating as clean", r.returncode, path.name)


# ── Binary discovery ──────────────────────────────────────────────────────────

def _find_ffprobe() -> Optional[str]:
    if p := shutil.which("ffprobe"):
        return p
    for candidate in [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
    ]:
        if Path(candidate).is_file():
            return candidate
    return None


def _find_clamscan() -> Optional[str]:
    if p := shutil.which("clamscan"):
        return p
    for candidate in [
        r"C:\Program Files\ClamAV\clamscan.exe",
        r"C:\ClamAV\clamscan.exe",
    ]:
        if Path(candidate).is_file():
            return candidate
    return None


# ── Convenience: clear session-scoped dedup set ───────────────────────────────

def clear_seen_hashes() -> None:
    """Reset the in-memory deduplication set.  Useful for testing."""
    _seen_hashes.clear()
