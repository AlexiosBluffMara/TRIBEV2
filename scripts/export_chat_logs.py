#!/usr/bin/env python
"""Export Claude Code session JSONL transcripts to readable markdown.

Walks `~/.claude/projects/<project-hash>/*.jsonl` and writes one markdown file
per session to `<out-root>/<project-hash>/<session-id>.md`. Idempotent: a
session is re-exported only if its source jsonl is newer than the existing
markdown output.

Default output root: `D:/TRIBEV2/archive/chatlogs/`.

Usage:
    python scripts/export_chat_logs.py                  # export all projects
    python scripts/export_chat_logs.py --project D--TRIBEV2
    python scripts/export_chat_logs.py --out D:/archive  --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_HOME", Path.home() / ".claude"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN {path.name} line {line_no}: {e}", file=sys.stderr)


def fmt_ts(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso


def render_content(content) -> str:
    """Flatten a message's content (which may be a string or a list of blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, indent=2)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "?")
            args = block.get("input", {})
            arg_str = json.dumps(args, ensure_ascii=False, indent=2)
            if len(arg_str) > 2000:
                arg_str = arg_str[:2000] + "\n... [truncated]"
            parts.append(f"\n**→ tool call: `{name}`**\n```json\n{arg_str}\n```")
        elif btype == "tool_result":
            out = block.get("content", "")
            if isinstance(out, list):
                out = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in out
                )
            out = str(out)
            if len(out) > 3000:
                out = out[:3000] + "\n... [truncated]"
            parts.append(f"\n**← tool result**\n```\n{out}\n```")
        elif btype == "thinking":
            parts.append(f"\n*[thinking]* {block.get('thinking', '')}")
        elif btype == "image":
            parts.append("\n*[image omitted]*")
        else:
            parts.append(f"\n*[{btype} block]*")
    return "\n".join(p for p in parts if p)


def render_session(jsonl_path: Path) -> str:
    header = [
        f"# Session `{jsonl_path.stem}`",
        "",
        f"- Source: `{jsonl_path}`",
        f"- Exported: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "---",
        "",
    ]
    body: list[str] = []
    first_ts: str | None = None
    last_ts: str | None = None
    n_user = n_assistant = n_tool = 0

    for rec in iter_jsonl(jsonl_path):
        rtype = rec.get("type") or rec.get("role")
        ts = rec.get("timestamp") or rec.get("created_at")
        if ts:
            first_ts = first_ts or ts
            last_ts = ts

        if rtype in ("user", "human"):
            n_user += 1
            msg = rec.get("message", rec)
            content = msg.get("content", rec.get("content", ""))
            body.append(f"## User — {fmt_ts(ts)}")
            body.append(render_content(content))
            body.append("")
        elif rtype == "assistant":
            n_assistant += 1
            msg = rec.get("message", rec)
            content = msg.get("content", rec.get("content", ""))
            body.append(f"## Assistant — {fmt_ts(ts)}")
            body.append(render_content(content))
            body.append("")
        elif rtype == "tool_use" or rtype == "tool_result":
            n_tool += 1
            body.append(f"### {rtype} — {fmt_ts(ts)}")
            body.append(render_content(rec.get("content", rec)))
            body.append("")
        elif rtype == "summary":
            body.append(f"> **Summary:** {rec.get('summary', '')}")
            body.append("")
        else:
            continue

    stats = [
        f"- Messages — user: {n_user}, assistant: {n_assistant}, tool: {n_tool}",
        f"- Started: {fmt_ts(first_ts) or 'unknown'}",
        f"- Ended:   {fmt_ts(last_ts) or 'unknown'}",
        "",
    ]
    return "\n".join(header + stats + ["---", ""] + body)


def export_project(project_dir: Path, out_root: Path, force: bool) -> tuple[int, int]:
    out_dir = out_root / project_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = skipped = 0
    for jsonl in sorted(project_dir.glob("*.jsonl")):
        out_md = out_dir / f"{jsonl.stem}.md"
        if not force and out_md.exists() and out_md.stat().st_mtime >= jsonl.stat().st_mtime:
            skipped += 1
            continue
        try:
            out_md.write_text(render_session(jsonl), encoding="utf-8")
            exported += 1
            print(f"  wrote {out_md.relative_to(out_root)}")
        except Exception as e:
            print(f"  ERROR {jsonl.name}: {e}", file=sys.stderr)
    return exported, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="D:/TRIBEV2/archive/chatlogs",
        help="Output root directory (default: D:/TRIBEV2/archive/chatlogs)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Export only this project hash (e.g. D--TRIBEV2). Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even when the markdown is newer than the jsonl.",
    )
    args = parser.parse_args()

    projects_root = claude_home() / "projects"
    if not projects_root.exists():
        print(f"No Claude projects directory at {projects_root}", file=sys.stderr)
        return 1

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    project_dirs = (
        [projects_root / args.project] if args.project else sorted(p for p in projects_root.iterdir() if p.is_dir())
    )

    total_exp = total_skip = 0
    for pd in project_dirs:
        if not pd.is_dir():
            print(f"skip {pd} (not a directory)", file=sys.stderr)
            continue
        print(f"[{pd.name}]")
        e, s = export_project(pd, out_root, args.force)
        total_exp += e
        total_skip += s

    print(f"\nDone. Exported {total_exp}, skipped {total_skip} (up-to-date).")
    print(f"Output: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
