# Account Consolidation — No-Takeout Strategy

**Goal:** Move email, photos, Drive files, passwords, and service-account ownership from multiple scattered accounts into one canonical business identity, before `soumitlahiri@philanthropytraders.com` Workspace terminates on **May 13, 2026**.

Takeout is off the table per user direction. Everything below uses MCP, IMAP, `rclone`, and direct app exports.

---

## Accounts in scope

| Account | Owns | Status |
|---------|------|--------|
| soumitlahiri@philanthropytraders.com | Domain `redteamkitchen.com` (maybe), current Google Cloud billing, possibly Kaggle | **Shutting down May 13** |
| lahirisoumit@gmail.com | Bizee LLC registration, Vyde tax account | Keep (or migrate to new canonical) |
| soumitty@gmail.com | Unknown — audit needed | Keep as secondary |
| (others TBD) | Audit needed | — |

**Decision needed from user:** which single Gmail becomes the canonical "business root" identity? Recommendation: **new dedicated account** like `redteamkitchen@gmail.com` or `alexiosbluffmara@gmail.com` so personal and business are cleanly separated.

---

## Phase 1 — Inventory (Days 1–2)

### Gmail MCP-driven service audit (per account)

For each account, run these MCP queries to build a service inventory:

```
search_threads: from:no-reply OR from:noreply  → auto-generated service emails
search_threads: subject:"welcome" OR subject:"verify your email"  → signups
search_threads: subject:"password" OR subject:"security alert"  → auth-related
search_threads: subject:"invoice" OR subject:"receipt" OR subject:"payment"  → paid services
search_threads: from:@google.com about:billing  → GCP / Workspace billing
```

Output: `docs/SERVICE_INVENTORY.csv` with columns:
- `service_name, login_email, last_activity_date, recovery_email_set, 2fa_method, has_payment_method, priority (high/med/low), migrated (Y/N)`

Priority scoring:
- **High:** payment methods, domain registrars, cloud billing, LLC/tax, bank-adjacent
- **Med:** dev tools (GitHub, Vercel), subscriptions with stored cards
- **Low:** newsletters, one-off signups, marketing lists

### Hardware inventory

User manually lists:
- Windows 11 RTX 5090 (primary)
- Pixel 9 Pro Fold
- Any other laptops / tablets / external drives
- Available storage capacity per device

---

## Phase 2 — Bulk Data Extraction (Days 3–7)

### Email — IMAP-to-MBOX (no Takeout)

1. Install **Thunderbird** on the 5090 box.
2. Add each Gmail as an IMAP account (enable IMAP in each Gmail's Settings → Forwarding and POP/IMAP, use App Password if 2FA is on).
3. Right-click each account → **Subscribe** → select all folders including `[Gmail]/All Mail`.
4. Let Thunderbird sync — this is the bottleneck. Expect 1–24 hrs depending on inbox size. Run in background.
5. Install **ImportExportTools NG** Thunderbird add-on.
6. Right-click each folder → **ImportExportTools NG → Export folder → MBOX**.
7. Archive resulting `.mbox` files:
   - Encrypt with `gpg --symmetric --cipher-algo AES256 inbox.mbox`
   - Upload to `gs://rtk-archive-cold/email/{account}/{date}.mbox.gpg`
   - Store GPG passphrase in Secret Manager AND printed in a safe location

### Google Drive — `rclone` direct

```bash
# Per account:
rclone config create soumit-old drive scope drive.readonly
rclone config create soumit-new drive scope drive

# Copy everything old → local archive
rclone copy soumit-old: D:/archive/drive-old/ --progress

# Copy local archive → new canonical account's drive
rclone copy D:/archive/drive-old/ soumit-new:imported-from-old/ --progress

# Also push to GCS cold archive
rclone copy D:/archive/drive-old/ gcs:rtk-archive-cold/drive-old/ --progress
```

### Google Photos — `gphotos-sync`

Google Photos is NOT in Drive by default. Use the open-source `gphotos-sync` tool:

```bash
pipx install gphotos-sync
gphotos-sync --new-token D:/archive/photos-soumitlahiri/
gphotos-sync --new-token D:/archive/photos-lahirisoumit/  # new auth flow
```

Re-upload to canonical account via Google Photos web uploader OR keep as local archive only.

### Passwords — export and consolidate

1. Visit `passwords.google.com` signed in as each account → **Export passwords** → CSV download. (This is NOT Takeout; it's the built-in password manager export.)
2. Import all CSVs into **one** consolidated password manager:
   - **Recommendation: Bitwarden (free, open source, self-hostable)** or **Proton Pass** (part of Proton ecosystem, aligns with sovereignty pitch)
   - Avoid: 1Password ($36/yr/person, lock-in) unless you already have a family plan
3. In the consolidated vault, tag each entry: `rotate-now`, `rotate-soon`, `kill` (for services to delete).
4. Rotate all `rotate-now` items (financial, email, domain registrars, cloud billing) **before May 13**.
5. `kill` items: sign in and delete the account at each service (don't just stop using them — closed accounts can't be phished).

---

## Phase 3 — Service Ownership Transfer (Days 5–13)

For each service on the inventory, action based on priority:

### High priority — transfer before May 13

| Service | Action |
|---------|--------|
| Domain registrar (where `redteamkitchen.com` lives — Squarespace? Google Domains/Squarespace sold?) | Transfer ownership to canonical Gmail |
| Google Cloud billing | Create new billing account under canonical Gmail, migrate project to it |
| LLC Bizee dashboard | Change contact email to canonical |
| Vyde (if not canceled yet) | Cancel, then export all tax docs to local archive |
| GitHub | Primary email change + verify |
| Stripe (if any) | Transfer ownership to canonical |
| Any bank/financial services | Update login email |

### Medium priority — can handle in May

- Dev tool subscriptions (Vercel, Railway, Fly.io, etc.)
- Discord developer account
- Kaggle
- HuggingFace
- LinkedIn (change primary email but keep profile)

### Low priority — just update or kill

- Newsletters, marketing lists
- Old e-commerce accounts
- Apps you signed into once and forgot

---

## Phase 4 — Consolidated Storage Architecture

### Public / Shared Cloud (GCP)

- **GCS bucket `rtk-archive-cold`** (Archive storage class, CMEK-encrypted): long-term email/drive/photos archive, pennies per GB/month
- **GCS bucket `rtk-datasets`** (Standard): fMRI datasets, Kaggle imports, model weights
- **GCS bucket `rtk-results`** (Standard): Jemma pipeline outputs, job results, web-facing assets
- **Secret Manager:** all API keys, DB credentials, service account JSONs — nothing in `.env` files committed

### Private / Local Backup (5090 box)

- **D:/archive** (your SSD): primary local copy, snapshotted weekly
- **Offsite copy:** Backblaze B2 or rsync.net for a third tier (~$5/month for 1TB at B2)
- **Restic or Kopia** for incremental encrypted backups

### Private VPC for Inference (GCP)

- Custom VPC `rtk-prod-vpc` in `us-central1`
- Subnets: `public-api` (Cloud Run egress), `private-inference` (Cloud Run GPU workers)
- Direct VPC Egress from Cloud Run (free, unlike Serverless VPC Access connector)
- All cross-service traffic stays in VPC
- Private Google Access enabled for Secret Manager + GCS calls

---

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 18 | Service inventory started, password managers set up, Thunderbird syncs running |
| Apr 22 | Service inventory complete, passwords exported and consolidated |
| Apr 25 | High-priority services all transferred |
| Apr 30 | Medium-priority services migrated |
| May 5 | Archives encrypted and uploaded to GCS cold storage |
| May 10 | Final sweep + verification (every high-priority service login tested from canonical email) |
| May 13 | `soumitlahiri@philanthropytraders.com` Workspace terminates — we should be **unaffected** |
| May 18 | Hackathon submissions (no dependency on old email) |

---

## Open questions (user decides)

1. **Canonical email identity for the business** — new Gmail, or keep `lahirisoumit@gmail.com`, or set up `soumit@redteamkitchen.com` via Google Workspace Business Starter ($7/mo)?
2. **Password manager** — Bitwarden (free/cheap, self-host option) or Proton Pass (bundled with Proton Mail)?
3. **Deepfake project** — flagged as "figure out later." Not in scope of this consolidation.

---

*Last updated: April 2026*
