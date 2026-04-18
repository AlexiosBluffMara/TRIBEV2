"""Live log watcher — tails logs/jemma.jsonl and calls Claude API on errors.

Usage:
    python -m bot.watch_logs              # tail from end of file
    python -m bot.watch_logs --from-start # replay full log history first

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY env var (or in .env)

Behaviour:
    - Follows jemma.jsonl in real time (polls every 0.5 s).
    - Batches ERROR/CRITICAL lines into a 10-second window.
    - Calls Claude claude-sonnet-4-6 with up to 40 lines of context.
    - Prints Claude's diagnosis + suggested fix.
    - In an interactive terminal, prompts whether to apply single-file patches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load .env if present (before importing config)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "jemma.jsonl"

_SYSTEM_PROMPT = """\
You are an expert Python debugger embedded in the Jemma Discord bot system.
Jemma is a Discord bot that runs a three-stage pipeline:
  Stage A: Gemma 4 E4B vision (cat gate) via Ollama
  Stage B: TRIBE v2 text-only inference
  Stage C: TRIBE v2 full multimodal (V-JEPA2 + wav2vec-BERT + Llama-3.2-3B)

You will be given a window of structured JSON log lines ending in one or more
ERROR or CRITICAL entries. Your job:

1. Identify the root cause in 2-3 sentences.
2. If the fix requires changing a file, output a PATCH block:
   --- PATCH: <relative/path/to/file.py> ---
   <exact old code to replace>
   === REPLACE WITH ===
   <exact new code>
   --- END PATCH ---
3. If the fix requires a user action (restart, update .env, etc.), say so
   explicitly under "ACTION REQUIRED:".
4. If the error is transient (network blip, OOM spike), say so.

Be concise. Only output what is needed to fix the problem.
"""


def _tail(path: Path, from_start: bool) -> None:
    """Follow a file, yielding new lines as they are written."""
    path.parent.mkdir(exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        if not from_start:
            fh.seek(0, 2)  # seek to end
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip()
            else:
                time.sleep(0.5)


def _call_claude(context_lines: list[str]) -> str:
    try:
        import anthropic
    except ImportError:
        return (
            "[watch_logs] anthropic package not installed.\n"
            "Run: pip install anthropic\n"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[watch_logs] ANTHROPIC_API_KEY not set — skipping Claude analysis.\n"

    client = anthropic.Anthropic(api_key=api_key)
    user_content = "Log context (newest last):\n\n" + "\n".join(context_lines)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text
    except Exception as exc:
        return f"[watch_logs] Claude API call failed: {exc}\n"


def _maybe_apply_patch(diagnosis: str, project_root: Path) -> None:
    """Parse PATCH blocks from Claude's response and offer to apply them."""
    if "--- PATCH:" not in diagnosis:
        return
    if not sys.stdin.isatty():
        return

    import re
    pattern = re.compile(
        r"--- PATCH: (?P<path>[^\n]+) ---\n(?P<old>.+?)=== REPLACE WITH ===\n(?P<new>.+?)--- END PATCH ---",
        re.DOTALL,
    )
    for m in pattern.finditer(diagnosis):
        rel_path = m.group("path").strip()
        old_code = m.group("old").rstrip()
        new_code = m.group("new").rstrip()
        target = project_root / rel_path

        if not target.exists():
            print(f"[watch_logs] Patch target not found: {rel_path}")
            continue

        current = target.read_text(encoding="utf-8")
        if old_code not in current:
            print(f"[watch_logs] Patch old-code not found in {rel_path} — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"PATCH for {rel_path}:")
        print(f"  REMOVE: {old_code[:120]!r}...")
        print(f"  INSERT: {new_code[:120]!r}...")
        answer = input("Apply this patch? [y/N] ").strip().lower()
        if answer == "y":
            target.write_text(
                current.replace(old_code, new_code, 1), encoding="utf-8"
            )
            print(f"[watch_logs] Patched {rel_path}.")
        else:
            print("[watch_logs] Skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Jemma log watcher")
    parser.add_argument(
        "--from-start", action="store_true",
        help="Replay full log history before following",
    )
    parser.add_argument(
        "--context", type=int, default=40,
        help="Lines of context to include in each Claude call (default 40)",
    )
    parser.add_argument(
        "--batch-window", type=float, default=10.0,
        help="Seconds to batch errors before calling Claude (default 10)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    print(f"[watch_logs] Watching {LOG_FILE}")
    print(f"[watch_logs] Ctrl-C to stop\n")

    context_buf: list[str] = []
    error_buf: list[str] = []
    last_error_time: float = 0.0

    for raw_line in _tail(LOG_FILE, from_start=args.from_start):
        # Keep a rolling context window
        context_buf.append(raw_line)
        if len(context_buf) > args.context:
            context_buf.pop(0)

        # Parse for level
        try:
            entry = json.loads(raw_line)
            level = entry.get("level", "")
        except json.JSONDecodeError:
            level = ""

        # Pretty-print to console
        ts = entry.get("ts", "") if level else ""
        msg = entry.get("msg", raw_line) if level else raw_line
        prefix = {
            "DEBUG": "\033[36m[DBG]\033[0m",
            "INFO": "\033[32m[INF]\033[0m",
            "WARNING": "\033[33m[WRN]\033[0m",
            "ERROR": "\033[31m[ERR]\033[0m",
            "CRITICAL": "\033[35m[CRT]\033[0m",
        }.get(level, "     ")
        print(f"{prefix} {ts} {msg}")

        if level in ("ERROR", "CRITICAL"):
            error_buf.append(raw_line)
            last_error_time = time.time()

        # Flush error batch once quiet for batch_window seconds
        if error_buf and (time.time() - last_error_time) >= args.batch_window:
            print(f"\n{'='*60}")
            print(f"[watch_logs] {len(error_buf)} error(s) detected — calling Claude...")
            diagnosis = _call_claude(list(context_buf))
            print(f"\n--- Claude diagnosis ---\n{diagnosis}\n{'='*60}\n")
            _maybe_apply_patch(diagnosis, project_root)
            error_buf.clear()


if __name__ == "__main__":
    main()
