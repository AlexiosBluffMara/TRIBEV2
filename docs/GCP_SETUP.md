# Google Cloud + Workspace CLI Setup — Red Team Kitchen

## Install order (Windows 11 on the 5090 box)

1. **gcloud CLI** (Google Cloud SDK)
2. **GAM7** (Google Workspace admin CLI)
3. **BigQuery CLI + gsutil** — bundled with gcloud, no extra install
4. **kaggle CLI** — for Kaggle dataset imports
5. **rclone** — cross-cloud file sync (already referenced in consolidation plan)

---

## 1. gcloud CLI

### Install (Windows, 5 minutes)

```powershell
# Download and run the installer:
# https://cloud.google.com/sdk/docs/install#windows
# Or via winget (preferred):
winget install Google.CloudSDK

# Restart shell, then:
gcloud version
```

### Initial auth

```bash
# Authenticate as canonical user
gcloud auth login
# Opens browser, sign in with soumitlahiri@philanthropytraders.com

# Authenticate for Application Default Credentials (Python SDK uses these)
gcloud auth application-default login

# Set default project
gcloud config set project rtk-prod-2026
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

### Create the project (one-time)

```bash
# Create project
gcloud projects create rtk-prod-2026 --name="Red Team Kitchen Production"

# Link a billing account (do this in console first, then:)
gcloud billing projects link rtk-prod-2026 \
    --billing-account=XXXXXX-XXXXXX-XXXXXX

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudtasks.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    cloudkms.googleapis.com \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com \
    compute.googleapis.com
```

### Storage + secrets scaffolding

```bash
# Buckets (matching docs/CONSOLIDATION.md)
gcloud storage buckets create gs://rtk-archive-cold \
    --location=us-central1 --storage-class=ARCHIVE --uniform-bucket-level-access
gcloud storage buckets create gs://rtk-datasets \
    --location=us-central1 --storage-class=STANDARD --uniform-bucket-level-access
gcloud storage buckets create gs://rtk-results \
    --location=us-central1 --storage-class=STANDARD --uniform-bucket-level-access

# CMEK key (optional; adds ~$2/mo but sovereignty signal)
gcloud kms keyrings create rtk-keyring --location=us-central1
gcloud kms keys create rtk-data-key \
    --keyring=rtk-keyring --location=us-central1 \
    --purpose=encryption

# Secret Manager — create first secret
echo -n "your-discord-bot-token" | gcloud secrets create discord-bot-token --data-file=-

# Retrieve later
gcloud secrets versions access latest --secret=discord-bot-token
```

### Billing alerts (do this before anything spends money)

```bash
# In console: Billing → Budgets & alerts → Create Budget
# Set: $100/month cap, alert at 50%, 90%, 100%
# Email target: soumitlahiri@philanthropytraders.com + lahirisoumit@gmail.com (legal backup)
```

---

## 2. GAM7 (Google Workspace admin CLI)

### Why it's needed

Workspace admin console is slow. GAM7 scripts bulk ops: adding domain aliases, reading audit logs, bulk user changes, service account management for cross-user automation.

### Install (Windows 11, 20 minutes first time)

```powershell
# Download MSI from https://github.com/GAM-team/GAM/releases/latest
# File: gam-7.XX.YY-windows-x86_64.msi
# Run installer — default install to C:\GAM7

# After install, set up API project + service account:
C:\GAM7\gam.exe config
# Follow prompts to:
# 1. Create GCP project for GAM (or use existing rtk-prod-2026)
# 2. Enable APIs
# 3. Create OAuth client for your admin account
# 4. Create service account for impersonation
# 5. Grant domain-wide delegation in Admin Console

# Detailed guide: https://github.com/taers232c/GAMADV-XTD3/wiki/How-to-Install-GAM7
```

### Useful commands after setup

```bash
# List all users
gam info domain

# Add user alias domain (Workspace admin CLI preferred over web for reproducibility)
gam create domain alias redteamkitchen.com type alias

# Create an alias for your user
gam update user soumitlahiri@philanthropytraders.com \
    add alias contact@redteamkitchen.com

# Audit: who has accessed what
gam report activities user soumitlahiri@philanthropytraders.com \
    todrive

# Bulk: export every user's drive file list
gam all users show filelist todrive
```

---

## 3. Kaggle CLI (for legal-dataset imports)

```bash
pip install kaggle
# Then: https://www.kaggle.com/settings/account → Create New API Token
# Save the downloaded kaggle.json to:
# Windows: C:\Users\soumi\.kaggle\kaggle.json
# Unix: ~/.kaggle/kaggle.json

chmod 600 ~/.kaggle/kaggle.json  # on WSL/Linux

# Test
kaggle competitions list

# Per dataset download → GCS
kaggle datasets download -d Cornell-University/arxiv -p D:/datasets/arxiv/
gcloud storage cp -r D:/datasets/arxiv/ gs://rtk-datasets/arxiv/
```

---

## 4. rclone (cross-cloud sync)

```powershell
winget install Rclone.Rclone

# Configure GCS remote
rclone config
# → n (new remote)
# → name: gcs
# → type: Google Cloud Storage
# → service_account_file: path to a GCS-only service account JSON
# → project_number: (from rtk-prod-2026)
# → location: us-central1

# Test
rclone ls gcs:rtk-datasets

# Sync local to GCS
rclone sync D:/datasets/ gcs:rtk-datasets/ --progress
```

---

## Where to get started — literally the next 5 commands

```powershell
# 1. Install gcloud
winget install Google.CloudSDK

# 2. Restart your shell, then auth
gcloud auth login

# 3. Create the project
gcloud projects create rtk-prod-2026 --name="Red Team Kitchen Production"

# 4. Link billing (do this in console: https://console.cloud.google.com/billing)

# 5. Enable core APIs
gcloud services enable run.googleapis.com storage.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com --project=rtk-prod-2026
```

After those 5 steps, you have a working GCP project. Everything else above layers on.

---

## What NOT to do

- **Don't store service account JSON files in your repo.** Use Secret Manager or gcloud Workload Identity Federation.
- **Don't use the default Compute Engine service account for Cloud Run.** Create a purpose-specific SA with least-privilege IAM.
- **Don't enable APIs you don't need.** Each enabled API is attack surface. Enable as you need them, not preemptively.
- **Don't skip the billing alert.** One misconfigured Cloud Run instance with GPU + autoscale can burn your budget overnight.

---

*Last updated: April 2026*
