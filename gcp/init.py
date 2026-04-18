"""
JemmaBrain — one-time GCP project initialization.

Run this after `gcloud auth login`:
    python gcp/init.py

Interactive wizard — asks for project ID, billing account, then
creates bucket, service accounts, secrets, Artifact Registry.
Does NOT require Terraform or any extra tools.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
_NOWWIN = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def _run(*args, check=True, capture=True):
    return subprocess.run(
        list(args), capture_output=capture, text=True,
        creationflags=_NOWWIN, check=check,
    )


def _gcloud(*args, check=True):
    gc = shutil.which('gcloud')
    if not gc:
        # Try common Windows install path
        candidates = [
            r'C:\Users\soumi\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd',
            r'C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd',
            r'C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd',
        ]
        for c in candidates:
            if Path(c).exists():
                gc = c
                break
    if not gc:
        print('ERROR: gcloud not found. Install from https://cloud.google.com/sdk/docs/install')
        sys.exit(1)
    return subprocess.run(
        [gc, *args], capture_output=True, text=True,
        creationflags=_NOWWIN, check=check,
    )


def _ask(prompt, default=''):
    val = input(f'{prompt} [{default}]: ').strip()
    return val if val else default


def _ok(msg):   print(f'  OK  {msg}')
def _skip(msg): print(f'  --  {msg} (already exists)')
def _err(msg):  print(f'  ERR {msg}')


def main():
    print('=' * 60)
    print('  JemmaBrain Google Cloud Setup')
    print('=' * 60)
    print()

    # Check gcloud auth
    r = _gcloud('auth', 'list', '--format=value(account)', '--filter=status:ACTIVE', check=False)
    if r.returncode != 0 or not r.stdout.strip():
        print('You are not authenticated. Run:')
        print('  gcloud auth login')
        print('  gcloud auth application-default login')
        sys.exit(1)

    account = r.stdout.strip().split('\n')[0]
    print(f'Authenticated as: {account}')
    print()

    # Collect inputs
    project_id = _ask('GCP Project ID (new or existing)', 'jemmabrain-prod')
    region     = _ask('Region', 'us-central1')
    zone       = f'{region}-a'
    bucket     = _ask('GCS bucket name', f'{project_id}-data')

    # Load .env values for secrets
    env_path = ROOT / '.env'
    env_vals = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, _, v = line.partition('=')
                env_vals[k.strip()] = v.strip().strip('"').strip("'")

    billing = _ask('Billing account ID (leave blank to skip)', env_vals.get('GCP_BILLING_ACCOUNT', ''))

    print()
    print(f'Project : {project_id}')
    print(f'Region  : {region}')
    print(f'Bucket  : gs://{bucket}')
    print()
    input('Press Enter to proceed (Ctrl+C to cancel)...')
    print()

    # ── 1. Create / set project ───────────────────────────────────────────────
    print('[1/7] Project...')
    r = _gcloud('projects', 'describe', project_id, check=False)
    if r.returncode != 0:
        _gcloud('projects', 'create', project_id, f'--name=JemmaBrain')
        _ok(f'Created project {project_id}')
    else:
        _skip(f'Project {project_id}')

    _gcloud('config', 'set', 'project', project_id)
    _gcloud('config', 'set', 'compute/region', region)
    _gcloud('config', 'set', 'compute/zone', zone)

    if billing:
        _gcloud('billing', 'projects', 'link', project_id, f'--billing-account={billing}', check=False)
        _ok('Billing linked')

    # ── 2. Enable APIs ────────────────────────────────────────────────────────
    print('[2/7] Enabling APIs...')
    apis = [
        'run.googleapis.com', 'compute.googleapis.com',
        'storage.googleapis.com', 'secretmanager.googleapis.com',
        'artifactregistry.googleapis.com', 'cloudbuild.googleapis.com',
        'iam.googleapis.com', 'logging.googleapis.com',
    ]
    _gcloud('services', 'enable', '--quiet', *apis)
    _ok('APIs enabled')

    # ── 3. GCS Bucket ─────────────────────────────────────────────────────────
    print('[3/7] GCS bucket...')
    r = _gcloud('storage', 'buckets', 'describe', f'gs://{bucket}', check=False)
    if r.returncode != 0:
        _gcloud('storage', 'buckets', 'create', f'gs://{bucket}',
                f'--location={region}', '--uniform-bucket-level-access',
                '--public-access-prevention')
        _ok(f'Created gs://{bucket}')
    else:
        _skip(f'gs://{bucket}')

    # Upload brain mesh if present
    mesh = ROOT / 'webapp' / 'public' / 'brain.glb'
    if mesh.exists():
        _gcloud('storage', 'cp', str(mesh), f'gs://{bucket}/mesh/brain.glb',
                '--cache-control=public, max-age=86400', check=False)
        nets = ROOT / 'webapp' / 'public' / 'networks.bin'
        if nets.exists():
            _gcloud('storage', 'cp', str(nets), f'gs://{bucket}/mesh/networks.bin',
                    '--cache-control=public, max-age=3600', check=False)
        _ok('Uploaded brain mesh to GCS')

    # ── 4. Service accounts ───────────────────────────────────────────────────
    print('[4/7] Service accounts...')
    sa_bot = f'{project_id}-bot'
    sa_web = f'{project_id}-web'

    for sa_name, display in [(sa_bot, 'JemmaBrain Bot'), (sa_web, 'JemmaBrain Web')]:
        r = _gcloud('iam', 'service-accounts', 'describe',
                    f'{sa_name}@{project_id}.iam.gserviceaccount.com', check=False)
        if r.returncode != 0:
            _gcloud('iam', 'service-accounts', 'create', sa_name, f'--display-name={display}')
            _ok(f'Created SA {sa_name}')
        else:
            _skip(f'SA {sa_name}')

    # Grant roles to bot SA
    bot_email = f'{sa_bot}@{project_id}.iam.gserviceaccount.com'
    for role in ['roles/storage.objectAdmin', 'roles/secretmanager.secretAccessor',
                 'roles/logging.logWriter', 'roles/run.invoker',
                 'roles/compute.instanceAdmin.v1']:
        _gcloud('projects', 'add-iam-policy-binding', project_id,
                f'--member=serviceAccount:{bot_email}', f'--role={role}', '--quiet', check=False)

    # Grant viewer roles to web SA
    web_email = f'{sa_web}@{project_id}.iam.gserviceaccount.com'
    _gcloud('projects', 'add-iam-policy-binding', project_id,
            f'--member=serviceAccount:{web_email}', '--role=roles/storage.objectViewer',
            '--quiet', check=False)
    _ok('IAM roles granted')

    # ── 5. Secrets ────────────────────────────────────────────────────────────
    print('[5/7] Secrets...')
    secrets = {
        'discord-bot-token': env_vals.get('DISCORD_BOT_TOKEN', 'REPLACE_ME'),
        'discord-guild-id':  env_vals.get('DISCORD_GUILD_ID',  'REPLACE_ME'),
        'gemma-api-key':     env_vals.get('GEMMA_API_KEY',     'REPLACE_ME'),
        'moonshot-api-key':  env_vals.get('MOONSHOT_API_KEY',  'REPLACE_ME'),
    }
    for name, value in secrets.items():
        r = _gcloud('secrets', 'describe', name, check=False)
        if r.returncode != 0:
            import io
            proc = subprocess.Popen(
                [shutil.which('gcloud') or 'gcloud', 'secrets', 'create', name,
                 '--replication-policy=automatic', '--data-file=-'],
                stdin=subprocess.PIPE, capture_output=True, text=True,
                creationflags=_NOWWIN,
            )
            proc.communicate(input=value)
            _ok(f'Created secret {name}')
        else:
            _skip(f'Secret {name}')

    # ── 6. Artifact Registry ──────────────────────────────────────────────────
    print('[6/7] Artifact Registry...')
    r = _gcloud('artifacts', 'repositories', 'describe', 'jemmabrain',
                f'--location={region}', check=False)
    if r.returncode != 0:
        _gcloud('artifacts', 'repositories', 'create', 'jemmabrain',
                '--repository-format=docker', f'--location={region}',
                '--description=JemmaBrain container images')
        _ok('Created jemmabrain artifact repo')
    else:
        _skip('jemmabrain artifact repo')

    registry = f'{region}-docker.pkg.dev/{project_id}/jemmabrain'

    # ── 7. Write .env additions ───────────────────────────────────────────────
    print('[7/7] Updating .env...')
    new_lines = [
        f'\n# Google Cloud (added by gcp/init.py)',
        f'GCP_PROJECT={project_id}',
        f'GCP_ZONE={zone}',
        f'GCS_BUCKET={bucket}',
        f'GCS_MESH_BASE=https://storage.googleapis.com/{bucket}/mesh',
        f'GCP_INFERENCE=0',  # enable with 1 when ready to use remote GPU
        f'GCP_REGISTRY={registry}',
        f'JEMMABRAIN_PUBLIC_URL=',  # fill in after Cloud Run deploy
    ]
    with open(env_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(new_lines) + '\n')
    _ok(f'Updated {env_path}')

    print()
    print('=' * 60)
    print(' GCP Setup Complete!')
    print('=' * 60)
    print()
    print(f' Project  : {project_id}')
    print(f' Bucket   : gs://{bucket}')
    print(f' Registry : {registry}')
    print()
    print(' Next steps:')
    print(f'  1. Build + deploy server:  gcloud builds submit . --config=gcp/cloudbuild-server.yaml')
    print(f'  2. Enable remote GPU:      Set GCP_INFERENCE=1 in .env')
    print(f'  3. Upload model weights:   gsutil cp models/tribe_v2.pt gs://{bucket}/models/')
    print(f'  4. Run a remote job:       bash gcp/run-inference.sh --video my_video.mp4')
    print()
    print(f' To get the Cloud Run URL after deploy:')
    print(f'   gcloud run services describe jemmabrain-server --region={region} --format="value(status.url)"')


if __name__ == '__main__':
    main()
