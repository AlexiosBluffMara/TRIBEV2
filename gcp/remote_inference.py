"""
JemmaBrain — Google Cloud remote inference client.

When the user submits a video via Discord and GCP_INFERENCE=1 is set,
this module launches a preemptible L4 GPU VM on GCP instead of running
the TRIBE v2 model locally.

Flow:
  1. Upload video to GCS  (gs://{bucket}/uploads/{job_id}_{filename})
  2. Launch g2-standard-8 + L4 VM with startup script
  3. Poll GCS for {job_id}_status.json sentinel (every 15s)
  4. Sync results back to local outputs/results/
  5. Return job_id for the Discord viewer link

Environment variables:
  GCP_INFERENCE=1         — enable remote inference
  GCS_BUCKET=my-bucket    — GCS bucket name
  GCP_PROJECT=my-project  — GCP project ID
  GCP_ZONE=us-central1-a  — compute zone
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GCP_INFERENCE = os.getenv('GCP_INFERENCE', '').strip() in ('1', 'true', 'yes')
GCS_BUCKET    = os.getenv('GCS_BUCKET', '').strip()
GCP_PROJECT   = os.getenv('GCP_PROJECT', '').strip()
GCP_ZONE      = os.getenv('GCP_ZONE', 'us-central1-a').strip()

_STARTUP_SCRIPT = ROOT / 'gcp' / 'startup-inference.sh'
_SA_NAME_TMPL   = '{project}-bot@{project}.iam.gserviceaccount.com'

_NOWWIN = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def _gcloud(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gcloud command and return CompletedProcess."""
    # Try both PATH and common install locations
    import shutil
    gcloud_bin = shutil.which('gcloud') or r'C:\Users\soumi\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud'
    cmd = [gcloud_bin, *args]
    return subprocess.run(
        cmd, capture_output=True, text=True,
        creationflags=_NOWWIN,
        check=check,
    )


def _gsutil(*args: str) -> subprocess.CompletedProcess:
    import shutil
    gsutil_bin = shutil.which('gsutil') or r'C:\Users\soumi\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil'
    return subprocess.run(
        [gsutil_bin, *args], capture_output=True, text=True,
        creationflags=_NOWWIN, check=False,
    )


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_video(video_path: Path, job_id: str) -> str:
    """Upload local video to GCS. Returns gs:// URI."""
    gcs_uri = f'gs://{GCS_BUCKET}/uploads/{job_id}_{video_path.name}'
    print(f'[gcp] Uploading {video_path.name} -> {gcs_uri}')
    _gsutil('cp', str(video_path), gcs_uri)
    return gcs_uri


# ── Launch VM ─────────────────────────────────────────────────────────────────

def launch_inference_vm(job_id: str, gcs_video: str, title: str = '') -> str:
    """
    Launch a preemptible g2-standard-8 (L4) VM to run TRIBE v2 inference.
    Returns the VM instance name.
    """
    instance_name = f'jemmabrain-inf-{job_id}'
    sa = _SA_NAME_TMPL.format(project=GCP_PROJECT)

    print(f'[gcp] Launching inference VM: {instance_name}')

    _gcloud(
        'compute', 'instances', 'create', instance_name,
        f'--zone={GCP_ZONE}',
        '--machine-type=g2-standard-8',
        '--accelerator=type=nvidia-l4,count=1',
        '--image-family=pytorch-latest-cu124',
        '--image-project=deeplearning-platform-release',
        '--boot-disk-size=100GB',
        '--boot-disk-type=pd-ssd',
        '--metadata=install-nvidia-driver=True',
        f'--metadata=bucket={GCS_BUCKET}',
        f'--metadata=job_id={job_id}',
        f'--metadata=video_gcs={gcs_video}',
        f'--metadata=title={title}',
        f'--metadata-from-file=startup-script={_STARTUP_SCRIPT}',
        f'--service-account={sa}',
        '--scopes=https://www.googleapis.com/auth/cloud-platform',
        '--provisioning-model=SPOT',
        '--instance-termination-action=DELETE',
        '--no-restart-on-failure',
        '--quiet',
    )

    print(f'[gcp] VM launched: {instance_name}')
    return instance_name


# ── Poll ──────────────────────────────────────────────────────────────────────

async def poll_job_completion(
    job_id: str,
    timeout: int = 3600,
    interval: int = 20,
) -> dict | None:
    """
    Async-poll GCS for the job completion sentinel.
    Returns the status dict when done, or None on timeout.
    """
    sentinel = f'gs://{GCS_BUCKET}/results/{job_id}_status.json'
    deadline = time.time() + timeout
    print(f'[gcp] Polling for job {job_id} (up to {timeout}s)...')

    while time.time() < deadline:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _gsutil('stat', sentinel)
        )
        if result.returncode == 0:
            # Sentinel exists — download it
            cat = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _gsutil('cat', sentinel)
            )
            try:
                status = json.loads(cat.stdout)
            except Exception:
                status = {'status': 'unknown'}
            print(f'[gcp] Job {job_id} complete: {status}')
            return status

        await asyncio.sleep(interval)

    print(f'[gcp] Job {job_id} timed out after {timeout}s')
    return None


# ── Sync results ──────────────────────────────────────────────────────────────

def sync_result_local(job_id: str) -> bool:
    """Download result files from GCS to local outputs/results/."""
    from bot.config import OUT_DIR
    dest = OUT_DIR / 'results'
    dest.mkdir(parents=True, exist_ok=True)

    for suffix in ('_meta.json', '_bold.bin'):
        src  = f'gs://{GCS_BUCKET}/results/{job_id}{suffix}'
        dst  = dest / f'{job_id}{suffix}'
        r = _gsutil('cp', src, str(dst))
        if r.returncode != 0:
            print(f'[gcp] WARNING: could not sync {src}: {r.stderr}')
            return False

    print(f'[gcp] Synced {job_id} to {dest}')
    return True


# ── High-level entry point ────────────────────────────────────────────────────

async def run_remote_inference(
    video_path: Path,
    job_id: str,
    title: str = '',
    progress_callback=None,
) -> str | None:
    """
    Full remote inference flow.  Returns job_id on success, None on failure.

    progress_callback(msg: str) is called with status updates (for Discord editing).
    """
    if not GCP_INFERENCE or not GCS_BUCKET or not GCP_PROJECT:
        return None

    def _report(msg: str):
        print(f'[gcp] {msg}')
        if progress_callback:
            try:
                asyncio.ensure_future(progress_callback(msg))
            except Exception:
                pass

    try:
        # 1. Upload video
        _report(f'Uploading video to Google Cloud Storage...')
        gcs_video = await asyncio.get_event_loop().run_in_executor(
            None, lambda: upload_video(video_path, job_id)
        )

        # 2. Launch VM
        _report(f'Launching GPU inference VM (NVIDIA L4, preemptible)...')
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: launch_inference_vm(job_id, gcs_video, title)
        )

        # 3. Poll for completion
        _report(f'Running TRIBE v2 pipeline on GCP (this takes 5-10 min)...')
        status = await poll_job_completion(job_id, timeout=1800, interval=20)

        if status is None:
            _report('Inference timed out after 30 minutes.')
            return None

        if status.get('status') != 'success':
            _report(f'Inference failed: {status}')
            return None

        # 4. Sync results
        _report('Syncing results from GCS to local storage...')
        ok = await asyncio.get_event_loop().run_in_executor(
            None, lambda: sync_result_local(job_id)
        )
        if not ok:
            _report('WARNING: Could not sync results locally. They are still in GCS.')

        _report(f'Done! Job {job_id} complete.')
        return job_id

    except Exception as exc:
        _report(f'Remote inference error: {exc}')
        return None
