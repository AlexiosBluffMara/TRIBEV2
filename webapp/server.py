"""
JemmaBrain FastAPI server — port 8765

Endpoints:
  GET  /mesh/brain.glb              → fsaverage5 pial GLTF (served once, cached)
  GET  /api/networks                → Int16Array binary of Yeo-7 vertex labels
  GET  /api/result/latest           → latest TRIBE result as JSON (for page reload)
  GET  /api/result/{job_id}         → specific saved result by job_id
  GET  /api/result/{job_id}/bold    → raw float32 BOLD binary for that result
  GET  /api/results                 → list of all saved results (newest first)
  POST /api/submit                  → accept uploaded media, queue pipeline job (streamed, validated)
  POST /api/submit-youtube          → accept YouTube URL, download via yt-dlp, queue
  WS   /ws/bold                     → push BOLD frames + analysis to all clients
  GET  /api/health                  → { status, gpu_vram_gb, n_clients }
"""

import asyncio
import collections
import json
import logging
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# ── Internal ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # D:\TRIBEV2
# Add project root so 'bot' is importable as a package (relative imports work).
# IMPORTANT: do NOT also add ROOT/bot — that puts bot/bot.py on the path as
# the top-level 'bot' module, shadowing the package and breaking relative imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger('jemmabrain.server')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

UPLOAD_DIR = ROOT / 'uploads'
MESH_PATH  = ROOT / 'webapp' / 'public' / 'brain.glb'
NET_PATH   = ROOT / 'webapp' / 'public' / 'networks.bin'   # Int16Array
DIST_DIR   = ROOT / 'webapp' / 'dist'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title='JemmaBrain', version='2.0')

# ── CORS — specific origins only (no wildcard in production) ──────────────────
_DEFAULT_CORS = ','.join([
    'http://localhost:5173',       # Vite dev server
    'http://localhost:8765',       # FastAPI itself (for dev tools)
    'http://127.0.0.1:5173',
    'http://127.0.0.1:8765',
    'https://brain.redteamkitchen.com',
    'https://redteamkitchen.com',
    'https://www.redteamkitchen.com',
])
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get('CORS_ORIGINS', _DEFAULT_CORS).split(',')
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization', 'X-Request-ID'],
    expose_headers=['X-Job-ID'],
    max_age=3600,
)

# ── Request size limit middleware ─────────────────────────────────────────────
_MAX_REQUEST_MB = int(os.environ.get('MAX_UPLOAD_MB', '500'))
_MAX_REQUEST_BYTES = _MAX_REQUEST_MB * 1024 * 1024

class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > _MAX_REQUEST_BYTES:
            return JSONResponse(
                {'error': f'Request body too large. Maximum {_MAX_REQUEST_MB} MB.'},
                status_code=413,
            )
        return await call_next(request)

app.add_middleware(MaxBodySizeMiddleware)

# ── IP-based rate limiting middleware ─────────────────────────────────────────
# Sliding window: counts requests per IP per minute.
# /api/submit has a tighter window (separate bucket).

_ip_general: dict[str, collections.deque]  = collections.defaultdict(lambda: collections.deque())
_ip_submit:  dict[str, collections.deque]  = collections.defaultdict(lambda: collections.deque())
_RATE_GENERAL_PER_MIN = int(os.environ.get('RATE_GENERAL_PER_MIN', '120'))
_RATE_SUBMIT_PER_MIN  = int(os.environ.get('RATE_SUBMIT_PER_MIN',  '6'))

def _rate_check(bucket: dict, ip: str, limit: int, window_s: float = 60.0) -> bool:
    """Return True if request is allowed; False if rate-limited. Mutates bucket."""
    now = time.monotonic()
    dq  = bucket[ip]
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True

class IPRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = (request.client.host if request.client else 'unknown')

        # Submit endpoint: stricter limit
        if request.url.path == '/api/submit':
            if not _rate_check(_ip_submit, ip, _RATE_SUBMIT_PER_MIN):
                return JSONResponse(
                    {'error': 'Upload rate limit exceeded.  Maximum 6 uploads per minute per IP.'},
                    status_code=429,
                    headers={'Retry-After': '60'},
                )

        # General API limit
        elif request.url.path.startswith('/api/') or request.url.path.startswith('/ws/'):
            if not _rate_check(_ip_general, ip, _RATE_GENERAL_PER_MIN):
                return JSONResponse(
                    {'error': 'Rate limit exceeded.  Please slow down.'},
                    status_code=429,
                    headers={'Retry-After': '30'},
                )

        return await call_next(request)

app.add_middleware(IPRateLimitMiddleware)

# Serve built frontend (Vite dist) if it exists
if DIST_DIR.exists():
    app.mount('/assets', StaticFiles(directory=str(DIST_DIR / 'assets')), name='assets')

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info('WS client connected (%d total)', len(self.active))

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        log.info('WS client disconnected (%d remaining)', len(self.active))

    async def broadcast_json(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_binary(self, data: bytes):
        dead = []
        for ws in self.active:
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_session_init(self, ws: WebSocket, session: dict):
        try:
            await ws.send_text(json.dumps({'type': 'session_init', **session}))
        except Exception:
            self.disconnect(ws)

manager = ConnectionManager()

# ── Latest result cache (in-memory, survives page refreshes) ─────────────────
_latest: dict = {}   # stores full result JSON

def cache_result(data: dict):
    global _latest
    _latest = data

# ── Pipeline job queue (shared with bot via file-based IPC or asyncio queue) ──
_job_queue: asyncio.Queue = asyncio.Queue()

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get('/mesh/brain.glb')
async def get_mesh():
    if not MESH_PATH.exists():
        return JSONResponse({'error': 'brain.glb not found — run scripts/export_brain_mesh.py'}, 404)
    return FileResponse(
        str(MESH_PATH),
        media_type='model/gltf-binary',
        headers={'Cache-Control': 'public, max-age=86400'},
    )


@app.get('/api/networks')
async def get_networks():
    """Return Yeo-7 network labels as raw Int16Array binary (n_vertices × 2 bytes)."""
    if not NET_PATH.exists():
        # Return zeros = all Unknown
        n_verts = 20484
        buf = (np.full(n_verts, -1, dtype=np.int16)).tobytes()
        return Response(content=buf, media_type='application/octet-stream')
    return Response(
        content=NET_PATH.read_bytes(),
        media_type='application/octet-stream',
        headers={'Cache-Control': 'public, max-age=3600'},
    )


@app.get('/api/results')
async def list_results_api():
    """Return list of all saved results (local + GCS), newest first."""
    try:
        gcs_bucket = os.getenv('GCS_BUCKET', '').strip()
        if gcs_bucket:
            from bot.gcs_store import list_results_unified
            return JSONResponse(list_results_unified())
        else:
            from bot.results_store import list_results
            return JSONResponse(list_results())
    except Exception as exc:
        import traceback
        log.error('list_results failed: %s\n%s', exc, traceback.format_exc())
        return JSONResponse([])


@app.get('/api/result/latest')
async def get_latest():
    """Return in-memory latest result, or load from disk if not yet cached."""
    if _latest:
        return JSONResponse(_latest)
    # Try loading the most recent saved result
    try:
        from bot.results_store import latest_result_id, load_result
        rid = latest_result_id()
        if rid:
            data = load_result(rid)
            if data:
                cache_result(data)
                return JSONResponse(data)
    except Exception:
        pass
    return JSONResponse({'error': 'no result yet'}, status_code=404)


@app.get('/api/result/{job_id}/bold')
async def get_result_bold(job_id: str):
    """
    Return raw float32 BOLD preds as binary (n_trs × n_verts × 4 bytes).
    Header: first 8 bytes = uint32 n_trs, uint32 n_verts (little-endian).
    Falls back to GCS if not found locally.
    """
    import struct as _struct

    # 1. Try local disk first
    try:
        from bot.results_store import RESULTS_DIR
        meta_path = RESULTS_DIR / f'{job_id}_meta.json'
        bold_path = RESULTS_DIR / f'{job_id}_bold.bin'
        if bold_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            header = _struct.pack('<II', meta['n_trs'], meta['n_verts'])
            return Response(
                content=header + bold_path.read_bytes(),
                media_type='application/octet-stream',
            )
    except Exception:
        pass

    # 2. Try GCS
    try:
        gcs_bucket = os.getenv('GCS_BUCKET', '').strip()
        if gcs_bucket:
            from bot.gcs_store import load_result_bold_bytes_gcs
            raw, n_trs, n_verts = load_result_bold_bytes_gcs(job_id)
            if raw is not None:
                header = _struct.pack('<II', n_trs, n_verts)
                return Response(
                    content=header + raw,
                    media_type='application/octet-stream',
                )
    except Exception as e:
        log.error('GCS bold fetch failed for %s: %s', job_id, e)

    return JSONResponse({'error': f'result {job_id!r} not found'}, status_code=404)


@app.get('/api/result/{job_id}')
async def get_result_by_id(job_id: str):
    """Return a specific saved result (metadata + bold_data as list)."""
    # Check in-memory cache first
    if _latest and _latest.get('job_id') == job_id:
        return JSONResponse(_latest)
    try:
        from bot.results_store import load_result
        data = load_result(job_id)
        if data is None:
            return JSONResponse({'error': f'result {job_id!r} not found'}, status_code=404)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get('/api/health')
async def health():
    info = {'status': 'ok', 'n_ws_clients': len(manager.active), 'ts': time.time()}
    try:
        import psutil
        info['cpu_pct'] = psutil.cpu_percent()
    except ImportError:
        pass
    try:
        import subprocess, sys as _sys
        _NOWWIN = subprocess.CREATE_NO_WINDOW if _sys.platform == 'win32' else 0
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=_NOWWIN,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(',')
            info['gpu_vram_used_gb'] = round(int(parts[0]) / 1024, 1)
            info['gpu_vram_total_gb'] = round(int(parts[1]) / 1024, 1)
            info['gpu_temp_c'] = int(parts[2].strip())
    except Exception:
        pass
    return JSONResponse(info)


@app.post('/api/submit')
async def submit_media(request: Request, file: UploadFile = File(...)):
    """
    Accept a media file upload and queue a TRIBE pipeline job.

    Security hardening:
    - Streaming upload (no full-file RAM load)
    - Magic byte verification
    - ffprobe integrity + codec-bomb detection
    - IP rate limiting (enforced by middleware)
    - Filename sanitization + path traversal prevention
    - Optional ClamAV scan
    """
    # Import validator (lazy to avoid circular imports on startup)
    try:
        from bot.file_validator import validate_and_save_streaming, ValidationError as FileValError
        _validator_available = True
    except ImportError:
        _validator_available = False

    if _validator_available:
        async def _chunk_iter():
            while chunk := await file.read(65_536):   # 64 KB chunks
                yield chunk

        try:
            validated = await validate_and_save_streaming(
                stream=_chunk_iter(),
                original_filename=file.filename or 'upload',
                upload_dir=UPLOAD_DIR,
                max_bytes=_MAX_REQUEST_BYTES,
            )
            save_path = validated.path
            extra_meta = {
                'sha256':      validated.hash[:16] + '…',
                'duration_s':  round(validated.duration_s, 1),
                'is_duplicate': validated.is_duplicate,
            }
        except FileValError as exc:
            log.warning('[submit] Rejected upload from %s: [%s] %s',
                        request.client.host if request.client else '?',
                        exc.code, exc.reason)
            return JSONResponse({'error': exc.reason, 'code': exc.code}, status_code=422)
    else:
        # Fallback (validator not installed — basic extension check only)
        log.warning('[submit] file_validator not available — using basic checks only')
        allowed = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        ext = Path(file.filename or '').suffix.lower()
        if ext not in allowed:
            return JSONResponse({'error': f'Unsupported file type: {ext}'}, status_code=400)
        content   = await file.read()
        save_path = UPLOAD_DIR / f'{int(time.time())}_{Path(file.filename or "upload").name}'
        save_path.write_bytes(content)
        extra_meta = {}

    job_id = f'web_{int(time.time())}'
    await _job_queue.put({'job_id': job_id, 'media_path': str(save_path)})

    # Kick off background pipeline task
    asyncio.create_task(run_pipeline_task(job_id, save_path))

    size_mb = round(save_path.stat().st_size / 1e6, 2) if save_path.exists() else 0
    log.info('[submit] queued job %s — %s (%.1f MB)', job_id, save_path.name, size_mb)

    return JSONResponse({
        'job_id': job_id,
        'status': 'queued',
        'size_mb': size_mb,
        **extra_meta,
    })


@app.post('/api/submit-youtube')
async def submit_youtube(request: Request):
    """
    Accept a YouTube video ID, download via yt-dlp, validate, and queue.
    Requires yt-dlp installed: pip install yt-dlp
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'error': 'Invalid JSON body.'}, status_code=400)

    video_id = (body.get('video_id') or '').strip()
    # Strict alphanumeric check (YouTube IDs are 11-char base64url)
    import re as _re
    if not _re.fullmatch(r'[A-Za-z0-9_\-]{6,16}', video_id):
        return JSONResponse({'error': 'Invalid YouTube video ID.'}, status_code=400)

    youtube_url = f'https://www.youtube.com/watch?v={video_id}'

    try:
        import shutil as _sh
        yt_dlp = _sh.which('yt-dlp') or _sh.which('yt_dlp')
        if not yt_dlp:
            return JSONResponse(
                {'error': 'yt-dlp is not installed on this server.  '
                          'Contact admin or upload the video file directly.'},
                status_code=503,
            )
    except Exception:
        return JSONResponse({'error': 'yt-dlp check failed.'}, status_code=503)

    job_id    = f'yt_{video_id}_{int(time.time())}'
    out_path  = UPLOAD_DIR / f'{job_id}.mp4'

    # Run yt-dlp in background — don't block the request
    async def _dl_and_queue():
        _NOWWIN_LOC = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        try:
            cmd = [
                yt_dlp,
                '--no-playlist',
                '--format', 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '--merge-output-format', 'mp4',
                '--output', str(out_path),
                '--max-filesize', '500m',
                '--',              # prevent URL injection
                youtube_url,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            if proc.returncode != 0 or not out_path.exists():
                log.error('[yt-dlp] failed for %s: %s', video_id, stderr.decode()[:300])
                return
            log.info('[yt-dlp] downloaded %s → %s', video_id, out_path.name)
            asyncio.create_task(run_pipeline_task(job_id, out_path))
        except asyncio.TimeoutError:
            log.error('[yt-dlp] timeout for %s', video_id)
        except Exception as exc:
            log.error('[yt-dlp] error: %s', exc)

    asyncio.create_task(_dl_and_queue())

    return JSONResponse({
        'job_id':    job_id,
        'video_id':  video_id,
        'status':    'downloading',
        'note':      'Download started. Results will appear in the viewer when complete.',
    })


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket('/ws/bold')
async def websocket_bold(ws: WebSocket):
    await manager.connect(ws)

    # Send latest session state if available
    if _latest:
        await manager.send_session_init(ws, {
            'n_trs':           _latest.get('n_trs', 0),
            'stimulus_title':  _latest.get('stimulus_title', ''),
            'n_vertices':      20484,
            'tr_seconds':      0.5,
        })
        # Send all cached BOLD frames
        bold = _latest.get('bold_data')
        if bold:
            await ws.send_text(json.dumps({
                'type':      'bold_all',
                'n_trs':     len(bold),
                'bold_data': bold,
            }))
        if _latest.get('analysis'):
            await ws.send_text(json.dumps({'type': 'analysis', **_latest['analysis']}))
        if _latest.get('narrations'):
            for tier, text in _latest['narrations'].items():
                await ws.send_text(json.dumps({'type': 'narration', 'tier': int(tier), 'text': text}))

    try:
        while True:
            # Keep connection alive; actual pushes come from broadcast_* calls
            await asyncio.sleep(30)
            await ws.send_text(json.dumps({'type': 'ping'}))
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ── Pipeline integration ──────────────────────────────────────────────────────

async def run_pipeline_task(job_id: str, media_path: Path):
    """
    Run TRIBE + BrainAnalysis + Gemma narrations in a background thread
    and stream results to all WebSocket clients as they arrive.
    """
    try:
        log.info('[pipeline] starting job %s for %s', job_id, media_path.name)

        await manager.broadcast_json({
            'type':    'stimulus',
            'title':   media_path.stem,
            'meta':    f'Job {job_id} — running pipeline…',
        })

        # Heavy imports (avoid slowing startup)
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        def _sync_pipeline():
            from bot import pipeline as _pipe
            from bot import analysis as _ana
            from bot import tiers    as _tiers

            result = _pipe.run_inference(str(media_path))
            ba     = _ana.analyse(result, harvard_oxford=True, juelich=True)
            nars   = _tiers.narrate_all_tiers(result, media_path.stem, ba.gemma_context())
            return result, ba, nars

        result, ba, nars = await loop.run_in_executor(executor, _sync_pipeline)

        # ── Session init ─────────────────────────────────────────────────────
        n_trs = result.preds.shape[0]
        await manager.broadcast_json({
            'type':            'session_init',
            'n_trs':           n_trs,
            'n_vertices':      result.preds.shape[1],
            'tr_seconds':      0.5,
            'stimulus_title':  media_path.stem,
        })

        # ── Stream BOLD frames as binary (4-byte idx + float32 array) ────────
        for i in range(n_trs):
            frame     = result.preds[i].astype(np.float32)
            idx_bytes = struct.pack('<I', i)           # uint32 LE
            payload   = idx_bytes + frame.tobytes()
            await manager.broadcast_binary(payload)
            await asyncio.sleep(0)                    # yield to event loop

        # ── Analysis JSON ────────────────────────────────────────────────────
        # Yeo-7 network activation (from network_means dict in BrainAnalysis)
        yeo_scores = {}
        if hasattr(ba, 'network_means') and ba.network_means:
            yeo_scores = {k: round(float(v), 4) for k, v in ba.network_means.items()}

        # Rich ROI list: [{name, z_score}] from Schaefer-400 top ROIs
        top_rois = []
        if hasattr(ba, 's400_roi_df') and hasattr(ba, 's400_top_rois') and ba.s400_top_rois:
            for roi_name in ba.s400_top_rois[:8]:
                if roi_name in ba.s400_roi_df.columns:
                    z = float(ba.s400_roi_df[roi_name].abs().mean())
                    top_rois.append({'name': roi_name.split('_', 2)[-1] if '_' in roi_name else roi_name,
                                     'full_name': roi_name, 'z_score': round(z, 3)})
        # Fallback to Harvard-Oxford
        if not top_rois and hasattr(ba, 'ho_top_rois') and ba.ho_top_rois:
            for roi_name in ba.ho_top_rois[:8]:
                if roi_name in ba.ho_roi_df.columns:
                    z = float(ba.ho_roi_df[roi_name].abs().mean())
                    top_rois.append({'name': roi_name.replace('HO-cort: ', '').replace('HO-sub:  ', ''),
                                     'z_score': round(z, 3)})

        # Cortex activation percentage
        n_verts      = result.preds.shape[1]
        peak_frame   = result.preds[result.peak_t] if result.peak_t else result.preds[0]
        active_verts = int(np.sum(np.abs(peak_frame) > 1.0))
        cortex_pct   = round(active_verts / n_verts * 100, 2)

        # Dominant network full name
        dom_net_code = getattr(ba, 'dominant_network', '')
        dom_net_full = {
            'Vis': 'Visual', 'SomMot': 'Somatomotor', 'DorsAttn': 'Dorsal Attention',
            'SalVentAttn': 'Salience / Ventral Attention', 'Limbic': 'Limbic',
            'Cont': 'Frontoparietal', 'Default': 'Default Mode',
        }.get(dom_net_code, dom_net_code or '—')

        analysis_payload = {
            'type':             'analysis',
            'peak_t':           float(ba.temporal.get('peak_s', 0)) if hasattr(ba, 'temporal') else 0.0,
            'dominant_network': dom_net_full,
            'cortex_pct':       cortex_pct,
            'yeo7_scores':      yeo_scores,
            'top_rois':         top_rois,
            'vertices_above_1sd': getattr(ba, 'vertices_above_1sd', 0),
            'vertices_above_2sd': getattr(ba, 'vertices_above_2sd', 0),
            'global_max_z':     round(float(getattr(ba, 'global_max_z', 0)), 3),
            'temporal':         getattr(ba, 'temporal', {}),
        }
        await manager.broadcast_json(analysis_payload)

        # ── Narrations by tier ───────────────────────────────────────────────
        for tier, text in nars.items():
            await manager.broadcast_json({'type': 'narration', 'tier': tier, 'text': text})
            await asyncio.sleep(0.05)

        # ── Cache for page reload ────────────────────────────────────────────
        cache_result({
            'stimulus_title': media_path.stem,
            'n_trs':          n_trs,
            'bold_data':      result.preds.tolist(),
            'analysis':       {k: v for k, v in analysis_payload.items() if k != 'type'},
            'narrations':     {str(k): v for k, v in nars.items()},
        })

        log.info('[pipeline] job %s complete — %d TRs streamed', job_id, n_trs)

    except Exception as e:
        log.exception('[pipeline] job %s failed: %s', job_id, e)
        await manager.broadcast_json({'type': 'error', 'message': str(e)})


# ── Public broadcaster (called by bot.py after a Discord job completes) ───────

async def push_result(result, ba, nars: dict, stimulus_title: str = 'Analysis'):
    """
    Called externally (e.g. from bot.py) to push a completed TRIBE result
    to all connected Three.js clients without going through the upload endpoint.
    """
    n_trs = result.preds.shape[0]

    await manager.broadcast_json({
        'type': 'session_init',
        'n_trs': n_trs,
        'n_vertices': result.preds.shape[1],
        'tr_seconds': 0.5,
        'stimulus_title': stimulus_title,
    })

    for i in range(n_trs):
        frame   = result.preds[i].astype(np.float32)
        payload = struct.pack('<I', i) + frame.tobytes()
        await manager.broadcast_binary(payload)
        await asyncio.sleep(0)

    yeo_scores = {}
    if hasattr(ba, 'network_means') and ba.network_means:
        yeo_scores = {k: round(float(v), 4) for k, v in ba.network_means.items()}

    top_rois = []
    if hasattr(ba, 's400_top_rois') and ba.s400_top_rois:
        for roi_name in ba.s400_top_rois[:8]:
            if hasattr(ba, 's400_roi_df') and roi_name in ba.s400_roi_df.columns:
                z = float(ba.s400_roi_df[roi_name].abs().mean())
                top_rois.append({'name': roi_name.split('_', 2)[-1] if '_' in roi_name else roi_name,
                                 'z_score': round(z, 3)})

    n_verts    = result.preds.shape[1]
    peak_frame = result.preds[result.peak_t] if result.peak_t else result.preds[0]
    cortex_pct = round(int(np.sum(np.abs(peak_frame) > 1.0)) / n_verts * 100, 2)
    dom_net_code = getattr(ba, 'dominant_network', '')
    dom_net_full = {
        'Vis': 'Visual', 'SomMot': 'Somatomotor', 'DorsAttn': 'Dorsal Attention',
        'SalVentAttn': 'Salience / Ventral Attention', 'Limbic': 'Limbic',
        'Cont': 'Frontoparietal', 'Default': 'Default Mode',
    }.get(dom_net_code, dom_net_code or '—')

    analysis_payload = {
        'type':             'analysis',
        'peak_t':           float(ba.temporal.get('peak_s', 0)) if hasattr(ba, 'temporal') else 0.0,
        'dominant_network': dom_net_full,
        'cortex_pct':       cortex_pct,
        'yeo7_scores':      yeo_scores,
        'top_rois':         top_rois,
        'vertices_above_1sd': getattr(ba, 'vertices_above_1sd', 0),
        'global_max_z':     round(float(getattr(ba, 'global_max_z', 0)), 3),
        'temporal':         getattr(ba, 'temporal', {}),
    }
    await manager.broadcast_json(analysis_payload)

    for tier, text in nars.items():
        await manager.broadcast_json({'type': 'narration', 'tier': tier, 'text': text})
        await asyncio.sleep(0.02)

    cache_result({
        'stimulus_title': stimulus_title,
        'n_trs':          n_trs,
        'bold_data':      result.preds.tolist(),
        'analysis':       {k: v for k, v in analysis_payload.items() if k != 'type'},
        'narrations':     {str(k): v for k, v in nars.items()},
    })


# ── Dev / standalone entrypoint ───────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'server:app',
        host='0.0.0.0',
        port=8765,
        reload=False,
        log_level='info',
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
