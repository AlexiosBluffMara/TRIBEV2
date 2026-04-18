# Kimi K2 vs Gemma 31B — Hermes Powering Decision

## Question
Should we buy a Kimi K2 API key to power our Nous Hermes submission, or power everything with Gemma 31B running on the 5090 + Cloud Run fallback?

## Answer: Gemma 31B. Skip Kimi for both hackathon submissions.

---

## Pricing (April 2026, confirmed)

| Model | Input per 1M tok | Output per 1M tok | With context cache hit |
|-------|------------------|-------------------|------------------------|
| Kimi K2 (0711 base) | $0.55 | $2.20 | ~$0.15 input |
| Kimi K2.5 | $0.60 | $2.50 | ~$0.15 input |
| Gemma 31B (local 5090) | $0 | $0 | N/A |
| Gemma 31B (Cloud Run L4, self-hosted) | ~$0.04 | ~$0.04 | N/A |

No Kimi subscription tier exists — it's pay-per-use. Budget is capped by your spend settings only.

## Cost model for a realistic Hermes agent session

Typical Hermes-style tool-use loop per task: ~15 LLM turns × ~4k input + 1k output per turn = 60k input + 15k output per task.

| 100 tasks/month | Kimi K2 | Gemma 31B local | Gemma 31B Cloud Run |
|-----------------|---------|-----------------|---------------------|
| Input cost | $3.30 | $0 | $0.24 |
| Output cost | $3.30 | $0 | $0.06 |
| **Total** | **$6.60** | **$0** | **$0.30** |

1,000 tasks/month = $66 Kimi vs $0 Gemma local.

## The hackathon judging argument

**Nous Research submission** specifically values **sovereignty, self-hosting, and agentic reasoning depth**. The narrative "our Hermes agent runs 100% on our own hardware with our own weights, no external API calls, full data control" is **stronger** than "we bought Kimi K2 credits."

**Gemma for Good submission** *requires* Gemma — Kimi would be disqualifying there.

## Tool-use benchmarks (context)

Kimi K2/K2.5 does beat Gemma 3/4 dense on some agentic tool-use benchmarks (τ-bench, ACEBench). The gap is real but typically 10–20 percentage points, not a categorical difference. For your use case — a bot that calls Jemma to analyze video clips and surfaces brain network activations — the tool graph is shallow. Gemma 31B handles shallow tool graphs fine.

## When Kimi would become worth adding

Revisit Kimi API **after** May 18 submission if:
- Nous Hermes reviewer feedback mentions agent planning depth
- We hit a specific tool-chain task where Gemma 31B fails and you can reproduce it 3+ times
- A paying customer asks for it specifically

If we add Kimi later: OpenRouter (`moonshotai/kimi-k2`) with a hard $25/mo spend cap is the safer on-ramp than direct Moonshot platform access.

## Decision locked

- **Hackathon Hermes bot:** Gemma 31B (local primary + Cloud Run L4 fallback)
- **Jemma brain narration:** Gemma 26B MoE (tiers 0–4) + Gemma 31B dense (tiers 5–6)
- **No external LLM APIs in the submission stack.** Full sovereignty pitch.

## Savings vs prior plan

Previous budget allocated $150 for Kimi API experimentation over 6 months. Reallocating:
- $100 → additional GCP buffer (more L4 GPU fallback headroom)
- $50 → NVIDIA DLI certification exam

---

*Last updated: April 2026*
