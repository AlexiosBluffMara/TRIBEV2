"""
Public URL tunnel manager for JemmaBrain.

Tries (in order):
  1. JEMMABRAIN_PUBLIC_URL in .env — use a manually set URL (e.g. Vercel, VPS)
  2. cloudflared (Cloudflare Tunnel) — free, persistent named subdomain
  3. ngrok — free tier with random URL

After a tunnel starts, the public base URL is stored in:
    outputs/.public_url      (read by bot.py to embed in Discord messages)

Usage:
    python -m bot.tunnel           # start tunnel, print URL, loop
    from bot.tunnel import get_public_url  # get current URL synchronously
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config

_NOWWIN = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
_URL_FILE = config.OUT_DIR / '.public_url'

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_url(url: str) -> None:
    _URL_FILE.write_text(url.strip(), encoding='utf-8')


def get_public_url() -> str | None:
    """Return the current public URL (from .env, file, or None)."""
    # 1. Explicit env override
    env_url = os.getenv('JEMMABRAIN_PUBLIC_URL', '').strip()
    if env_url:
        return env_url.rstrip('/')
    # 2. Written by tunnel process
    if _URL_FILE.exists():
        url = _URL_FILE.read_text(encoding='utf-8').strip()
        if url:
            return url.rstrip('/')
    return None


def viewer_url(job_id: str, base: str | None = None) -> str:
    """Return the browser URL for a specific result."""
    base = base or get_public_url() or 'http://localhost:5173'
    return f'{base}/?r={job_id}'


def latest_viewer_url() -> str | None:
    """Return viewer URL for the most recently saved result."""
    from .results_store import latest_result_id
    rid = latest_result_id()
    if not rid:
        return None
    return viewer_url(rid)


# ── Cloudflare tunnel ─────────────────────────────────────────────────────────

def _find_cloudflared() -> str | None:
    """Return path to cloudflared binary if found."""
    import shutil
    return shutil.which('cloudflared')


async def start_cloudflare_tunnel(port: int = 5173) -> str | None:
    """
    Start 'cloudflared tunnel --url http://localhost:<port>' and parse the URL.
    Returns the public https:// URL or None on failure.
    """
    cf = _find_cloudflared()
    if not cf:
        return None

    named_tunnel = os.getenv('CF_TUNNEL_NAME', '').strip()

    if named_tunnel:
        # Named tunnel: requires prior `cloudflared tunnel create <name>`
        # and ~/.cloudflared/config.yml
        cmd = [cf, 'tunnel', 'run', named_tunnel]
    else:
        # Quick tunnel (random trycloudflare.com subdomain)
        cmd = [cf, 'tunnel', '--url', f'http://localhost:{port}']

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        creationflags=_NOWWIN,
    )

    url_pattern = re.compile(r'https://[a-z0-9\-]+\.trycloudflare\.com|https://[a-z0-9\-]+\.cfargotunnel\.com')
    timeout = 30
    start   = time.time()
    url     = None

    while time.time() - start < timeout:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
        text = line.decode(errors='replace').strip()
        if text:
            m = url_pattern.search(text)
            if m:
                url = m.group(0)
                break

    if url:
        _write_url(url)
        print(f'[tunnel] Cloudflare URL: {url}')
    else:
        proc.kill()
    return url


# ── ngrok ─────────────────────────────────────────────────────────────────────

def _find_ngrok() -> str | None:
    import shutil
    return shutil.which('ngrok')


async def start_ngrok_tunnel(port: int = 5173) -> str | None:
    """
    Start 'ngrok http <port>' and parse the public URL from ngrok's local API.
    Supports ngrok v2 and v3. Returns the public https:// URL or None.
    """
    ngrok = _find_ngrok()
    if not ngrok:
        return None

    proc = await asyncio.create_subprocess_exec(
        ngrok, 'http', str(port),
        '--log', 'stdout',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        creationflags=_NOWWIN,
    )

    # Give ngrok up to 10s to start
    url_pattern = re.compile(r'https://[a-z0-9\-]+\.ngrok[-a-z.io]*\.app|https://[a-z0-9\-]+\.ngrok\.io')
    start = time.time()

    async def _read_ngrok_url():
        while time.time() - start < 10:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=2)
                text = line.decode(errors='replace').strip()
                m    = url_pattern.search(text)
                if m:
                    return m.group(0)
                # ngrok v3 uses the web interface at :4040
                if 'started tunnel' in text.lower() or 'url=' in text.lower():
                    break
            except asyncio.TimeoutError:
                break
        return None

    url = await _read_ngrok_url()

    # Fallback: poll ngrok's local API
    if not url:
        await asyncio.sleep(2)
        try:
            import urllib.request, json as _json
            for api_url in ('http://localhost:4040/api/tunnels', 'http://127.0.0.1:4040/api/tunnels'):
                try:
                    with urllib.request.urlopen(api_url, timeout=3) as r:
                        data = _json.loads(r.read())
                    for t in data.get('tunnels', []):
                        pu = t.get('public_url', '')
                        if pu.startswith('https://'):
                            url = pu
                            break
                    if url:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if url:
        _write_url(url)
        print(f'[tunnel] ngrok URL: {url}')
    else:
        try:
            proc.kill()
        except Exception:
            pass

    return url


# ── Auto-start (tries cloudflare, then ngrok) ─────────────────────────────────

async def start_tunnel(port: int = 5173) -> str | None:
    """Try to start the best available tunnel. Returns public URL or None."""
    # Explicit override
    env_url = os.getenv('JEMMABRAIN_PUBLIC_URL', '').strip()
    if env_url:
        _write_url(env_url)
        print(f'[tunnel] Using explicit URL: {env_url}')
        return env_url

    url = await start_cloudflare_tunnel(port)
    if url:
        return url

    url = await start_ngrok_tunnel(port)
    if url:
        return url

    print('[tunnel] No tunnel available. Using http://localhost:5173 (local only).')
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    async def _main():
        url = await start_tunnel()
        if url:
            print(f'[tunnel] Public URL: {url}')
            print('[tunnel] Press Ctrl+C to stop')
            try:
                while True:
                    await asyncio.sleep(60)
            except KeyboardInterrupt:
                pass
        else:
            print('[tunnel] Could not establish a tunnel.')

    asyncio.run(_main())
