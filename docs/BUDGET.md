# Red Team Kitchen — Budget & Cost Model

## Phase 1: Local-First (Now → Month 2) — ~$40/month

| Line Item | Monthly | Notes |
|---|---|---|
| Squarespace Business | $23 | Custom code injection, forms, analytics |
| Cloudflare (tunnel) | $0 | Free forever for tunnels |
| Domain (redteamkitchen.com) | $1 | ~$12/year amortized |
| GCP Cloud Tasks | $1 | 1M tasks/month free, overflow only |
| GCP Cloud Run (API) | $5 | 1 always-on instance, minimal traffic |
| GCS Storage | $0.25 | ~10 GB results + brain mesh CDN |
| Electricity (RTX 5090) | $15 | ~300W × 24h × 30d × $0.07/kWh |
| **TOTAL** | **~$45/mo** | |

## Phase 2: Internet-Exposed (Month 2–4) — ~$90/month

| Line Item | Monthly | Notes |
|---|---|---|
| Phase 1 costs | $45 | As above |
| Sendgrid (email notifications) | $0 | Free 100/day |
| Cloudflare Pro (analytics, WAF) | $20 | DDoS protection, bot filtering |
| GCS CDN (brain.glb serving) | $2 | ~50 GB egress |
| Monitoring (Cloud Monitoring) | $0 | Free tier |
| Stripe (payment processing) | $0 | % only, no monthly fee |
| **TOTAL** | **~$67/mo** | |

## Phase 3: Hybrid Cloud (Month 4–6) — ~$150/month (before grants)

| Line Item | Monthly | Notes |
|---|---|---|
| Phase 2 costs | $67 | As above |
| GCP L4 GPU VM (fallback) | $40 | ~50 overflow jobs/month × $0.80 |
| GCP Batch (batch inference) | $15 | Night-window jobs at spot pricing |
| Vertex AI (future) | $25 | When migrating away from Ollama |
| **TOTAL** | **~$147/mo** | |

## Revenue Model

### Free Tier (Public)
- 1 job/hour, E4B model, basic narration
- Goal: Drive volume, social sharing, word of mouth

### Verified Tier ($5 one-time or free with ISU email)
- 4 jobs/hour, 26B MoE model, deep narration
- Verified via Discord phone link OR academic email

### Researcher API ($49/month)
- 20 jobs/hour, 31B expert model, full 7-tier narration
- REST API access with job webhooks
- Priority queue position
- Target: Academic institutions, media researchers, documentary producers

### Enterprise ($500/month)
- Unlimited, private results, custom narration training
- SLA, dedicated support
- Target: ISU department license, hospitals, content studios

### Grant/Donation
- NSF SBIR Phase I: ~$275K (single source, 6-month runway)
- NVIDIA Inception credits: reduces compute cost to $0
- Google for Startups: $100K GCP (eliminates cloud cost for 1 year)

## Break-Even Analysis

| Users/month | Avg Revenue | Monthly Revenue | Cost | Profit |
|---|---|---|---|---|
| 100 (all free) | $0 | $0 | $67 | -$67 |
| 100 (10% Researcher) | $49 | $490 | $150 | $340 |
| 500 (5% Researcher) | $49 | $1,225 | $200 | $1,025 |
| ISU department license | $500 | $500 | $150 | $350 |

**Break-even:** 2 Researcher subscribers or 1 small institutional license.

## One-Time Startup Costs

| Item | Cost | Notes |
|---|---|---|
| RTX 5090 GPU | Already owned | — |
| Squarespace domain transfer | $0 | Already on Squarespace |
| Legal (LLC operating agreement update) | $150 | One-time |
| NVIDIA Inception application | $0 | Free |
| Kaggle hackathon entry | $0 | Free |
| NSF SBIR proposal writing | $0 | DIY or ISU faculty co-PI |
| Google Cloud Professional ML Exam | $200 | LinkedIn certification |
| **TOTAL** | **~$350** | |

## 12-Month Financial Projection

| Month | Revenue | Expenses | Notes |
|---|---|---|---|
| 1–2 | $0 | $90 | Building, Cloudflare tunnel live |
| 3 | $0 | $90 | ISU pilot launch |
| 4 | $490 | $150 | First 10 Researcher subscribers |
| 5–6 | $980 | $150 | Growth, NSF application in |
| 7–8 | $1,500 | $200 | ISU department license |
| 9–12 | $2,500 | $250 | Enterprise growth, grant pending |
| **Year 1 Total** | **~$14K** | **~$1.5K** | ~$12.5K net (pre-grant) |

If NSF Phase I awarded (Month 11): +$275K one-time.  
If NVIDIA Inception: -$150K compute = essentially $0 cloud costs for 12+ months.
