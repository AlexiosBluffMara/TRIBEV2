"""
JemmaBrain Visualization Launcher
===================================
Starts the full stack for the Three.js cortical visualizer:

  1.  scripts/export_brain_mesh.py  — exports brain.glb + networks.bin (once)
  2.  webapp/server.py (uvicorn)    — FastAPI at http://localhost:8765
  3.  npm run dev (vite)             — Three.js dev server at http://localhost:5173

Usage:
    python start_viz.py            # start full stack
    python start_viz.py --api-only # just the FastAPI server (no Vite)
    python start_viz.py --export   # just re-export brain mesh then exit

Press Ctrl+C to stop all processes.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).parent
WEBAPP    = ROOT / 'webapp'

# Use the TRIBEV2 project venv if it exists (has nilearn, torch, etc.)
# The venv can live at C:\Users\...\TRIBEV2\.venv  OR  D:\TRIBEV2\.venv
_VENV_CANDIDATES = [
    ROOT / '.venv' / 'Scripts' / 'python.exe',   # D:\TRIBEV2\.venv
    Path.home() / 'TRIBEV2' / '.venv' / 'Scripts' / 'python.exe',  # C:\Users\...\TRIBEV2\.venv
    Path(sys.executable),                          # fallback: current Python
]
VENV_PYTHON = next((p for p in _VENV_CANDIDATES if p.exists()), Path(sys.executable))

# Windows subprocess flags
_NOWWIN = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def banner(msg: str):
    print(f'\n\033[1;34m[JemmaBrain]\033[0m {msg}')


def run_export():
    banner('Exporting brain mesh…')
    result = subprocess.run(
        [str(VENV_PYTHON), 'scripts/export_brain_mesh.py'],
        cwd=str(ROOT),
        creationflags=_NOWWIN,
    )
    if result.returncode != 0:
        print('\033[1;33m[WARN]\033[0m Mesh export failed — using fallback (all-unknown networks)')


def start_api() -> subprocess.Popen:
    banner('Starting FastAPI server at http://localhost:8765')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'bot') + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.Popen(
        [
            str(VENV_PYTHON), '-m', 'uvicorn',
            'webapp.server:app',
            '--host', '0.0.0.0',
            '--port', '8765',
            '--reload',
            '--reload-dir', str(WEBAPP),
            '--log-level', 'info',
        ],
        cwd=str(ROOT),
        env=env,
        creationflags=_NOWWIN,
    )
    return proc


def start_vite() -> subprocess.Popen:
    banner('Starting Vite dev server at http://localhost:5173')
    # Detect node/npm
    npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
    node_modules = WEBAPP / 'node_modules'
    if not node_modules.exists():
        banner('Running npm install first…')
        subprocess.run([npm_cmd, 'install'], cwd=str(WEBAPP), check=True)

    proc = subprocess.Popen(
        [npm_cmd, 'run', 'dev'],
        cwd=str(WEBAPP),
        creationflags=_NOWWIN,
    )
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-only', action='store_true')
    ap.add_argument('--export',   action='store_true')
    ap.add_argument('--no-export', action='store_true', help='Skip brain mesh export step')
    args = ap.parse_args()

    if args.export:
        run_export()
        return

    # Check if mesh files exist
    glb_path = WEBAPP / 'public' / 'brain.glb'
    if not glb_path.exists() and not args.no_export:
        run_export()
    elif args.no_export:
        banner('Skipping mesh export (--no-export)')

    procs: list[subprocess.Popen] = []

    try:
        api_proc = start_api()
        procs.append(api_proc)

        # Give API a moment to start
        time.sleep(2)

        if not args.api_only:
            vite_proc = start_vite()
            procs.append(vite_proc)
            time.sleep(1)

        banner('Stack running — press Ctrl+C to stop')
        print()
        print('  🌐  Visualizer: http://localhost:5173')
        print('  ⚡  API:        http://localhost:8765')
        print('  🔌  WebSocket:  ws://localhost:8765/ws/bold')
        print('  🩺  Health:     http://localhost:8765/api/health')
        print()

        # Wait until killed
        for proc in procs:
            proc.wait()

    except KeyboardInterrupt:
        banner('Shutting down…')
    finally:
        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        banner('Stopped.')


if __name__ == '__main__':
    main()
