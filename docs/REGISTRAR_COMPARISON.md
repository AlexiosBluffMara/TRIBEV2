# Domain Registrar Comparison — Where redteamkitchen.com Should Live

## TL;DR

**Cloudflare Registrar** is the right choice for Red Team Kitchen. Porkbun is the only serious alternative worth considering. Everything else is either more expensive or weaker on privacy/integration.

## Criteria

1. **Cost** — at-cost preferred (ICANN wholesale + ~$0.18)
2. **WHOIS privacy** — free, always-on
3. **DNS included and fast** — Cloudflare anycast or equivalent
4. **Tunnel / edge integration** — Cloudflare Tunnel is already in ROADMAP.md
5. **Sovereignty + business continuity** — account identity you control forever
6. **API / scriptable** — gcloud-style CLI workflows matter

## Comparison

| Registrar | Cost/yr (.com) | WHOIS privacy | DNS | Tunnel/edge | API | Notes |
|-----------|----------------|---------------|-----|-------------|-----|-------|
| **Cloudflare** | $9.77 (at-cost) | Free, always | Free, fast anycast | Free Cloudflare Tunnel | Yes, excellent | Only accepts transfers (no new registration) |
| **Porkbun** | $9.73 | Free | Free | None (bring your own) | Yes, clean | Cheapest, dev-friendly, no lock-in |
| **Namecheap** | $14.98 (first yr promo ~$9) | Free | Basic included | None | Limited | Common default, not best |
| **Squarespace** (current) | ~$20 | Paid add-on | Included with website | None | Limited | Fine but expensive, ties to Squarespace account |
| **Google Domains** | N/A | — | — | — | — | Sold to Squarespace, migrated |
| **AWS Route 53** | $14+ | Free | Strong | Integrates with CloudFront | Yes | AWS lock-in, more expensive |
| **Gandi.net** | $18 | Free | Basic | None | Yes | EU jurisdiction, older UI |
| **Hover** | $18 | Free | Basic | None | No | Nice UI, overpriced |

## Privacy deep dive

- **Cloudflare**: WHOIS obscured to `DATA REDACTED` by default; Cloudflare acts as intermediary for legal process. US jurisdiction.
- **Porkbun**: Same WHOIS redaction behavior. US jurisdiction. Slightly more contactable by abuse reports.
- **Gandi**: EU jurisdiction (France) — better privacy regime but more exposure to EU enforcement if EU law is relevant.
- **Namecheap / Squarespace**: WHOIS privacy works but the registrar's accounts themselves are less hardened.

## AI / cloud integration

Cloudflare has direct integrations for:
- **Workers AI** (edge inference, useful if you eventually want to run tiny models at edge)
- **AI Gateway** (rate-limit / cache LLM API calls — would matter if you ever reverse course on Kimi)
- **Vectorize** (managed vector DB)
- **R2** (S3-compatible object storage, no egress fees — a real alternative to GCS for some assets)

**None of this locks you to Cloudflare inferencing** — you can use these selectively while keeping your primary GCP stack. But it's a nice backup / multi-cloud position if GCP ever has an outage.

Porkbun does not offer edge compute.

## Recommended setup

1. **Registrar: Cloudflare** — transfer `redteamkitchen.com` from Squarespace (~$9.77/yr, locks in at transfer)
2. **DNS: Cloudflare** — included free, already supports Cloudflare Tunnel
3. **Tunnel: Cloudflare Tunnel** — per ROADMAP.md, `brain.redteamkitchen.com` → RTX 5090
4. **Edge static assets: Cloudflare R2** (optional) — for large .glb brain meshes if GCS egress gets expensive
5. **Primary compute + data: GCP** — Cloud Run, GCS, Secret Manager, Vertex AI (when needed)

This gives you: cheap/private registrar, edge CDN/tunnel, no vendor lock on the primary stack.

## Why NOT to do dual-cloud primary

Your $3K budget doesn't support running production infra across two clouds. Your hackathon pitch is stronger with a clean single-cloud story. Keep GCP as the single compute+storage home; use Cloudflare only for the commoditized edge layer (registrar, DNS, tunnel, optional R2).

## Migration steps (when ready)

1. Log into Squarespace as `soumitty@gmail.com`
2. Domains → `redteamkitchen.com` → Advanced → Unlock transfer, copy EPP/auth code
3. Create Cloudflare account under your canonical email (`soumitlahiri@philanthropytraders.com`)
4. Cloudflare Dashboard → Domain Registration → Transfer → enter domain + EPP code + pay $9.77
5. Wait 5–7 days for ICANN transfer to complete (DNS stays active throughout — no downtime)
6. Post-transfer: add MX records pointing to Google Workspace (for email), verification TXT records, the `brain` CNAME for Cloudflare Tunnel

---

*Last updated: April 2026*
