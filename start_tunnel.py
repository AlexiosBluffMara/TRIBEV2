"""
JemmaBrain public URL tunnel
===============================
Starts an ngrok (or cloudflare) tunnel so the Three.js brain viewer is
accessible from any browser — including from Discord embeds.

Usage:
    python start_tunnel.py           # tunnel Vite dev server (:5173)
    python start_tunnel.py --port 5173
    python start_tunnel.py --prod    # tunnel the built app served by FastAPI (:8765)

The public URL is written to outputs/.public_url and picked up automatically
by the Discord bot when it posts the 3D viewer link.

Press Ctrl+C to stop.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'bot'))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5173, help='Local port to tunnel (default 5173)')
    ap.add_argument('--prod', action='store_true', help='Tunnel port 8765 (FastAPI/production)')
    args = ap.parse_args()

    port = 8765 if args.prod else args.port

    # Import here so .env is already loaded
    from bot.tunnel import start_tunnel, get_public_url, _URL_FILE

    # Check if URL already set via env
    env_url = os.getenv('JEMMABRAIN_PUBLIC_URL', '').strip()
    if env_url:
        print(f'[tunnel] Using JEMMABRAIN_PUBLIC_URL from .env: {env_url}')
        _URL_FILE.write_text(env_url, encoding='utf-8')
        print('[tunnel] Written to outputs/.public_url')
        print('[tunnel] The Discord bot will use this URL for 3D viewer links.')
        return

    print(f'[tunnel] Starting tunnel for port {port}...')
    url = await start_tunnel(port)

    if not url:
        print('[tunnel] Could not start a tunnel.')
        print()
        print('Options:')
        print('  1. Set JEMMABRAIN_PUBLIC_URL=https://your-domain.com in .env')
        print('  2. Install ngrok:  winget install ngrok  (then ngrok config add-authtoken <token>)')
        print('  3. Install cloudflared:  winget install Cloudflare.cloudflared')
        sys.exit(1)

    print()
    print('=' * 60)
    print(f'  PUBLIC URL: {url}')
    print('=' * 60)
    print()
    print('Discord embeds will now link to this URL.')
    print('Share it with anyone — opens a live 3D brain viewer in their browser.')
    print()
    print('Keeping tunnel alive... Press Ctrl+C to stop.')
    print()

    try:
        while True:
            await asyncio.sleep(10)
            # Refresh URL file in case it changed
            current = get_public_url()
            if current and current != url:
                print(f'[tunnel] URL updated: {current}')
                url = current
    except KeyboardInterrupt:
        pass
    finally:
        # Clear the URL file so bot doesn't try to use a dead tunnel
        if _URL_FILE.exists():
            _URL_FILE.unlink()
        print('[tunnel] Stopped. URL file cleared.')


if __name__ == '__main__':
    asyncio.run(main())
