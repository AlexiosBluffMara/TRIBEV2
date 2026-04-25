# Three-way eval: base vs v2 vs v3 (n=30 paired samples)

Snapshot: `D:\TRIBEV2\outputs\paper\eval_stats_three_way_gemma4\eval_gemma4_three_way_1776674142`

## Style-transfer rates (binary)

| metric | base | v2 | v3 | Δ(v2-base) | Δ(v3-base) | Δ(v3-v2) |
|---|---:|---:|---:|---:|---:|---:|
| opens_with_template | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | +0.000 |
| has_tribe_disclaimer | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | +0.000 |
| has_not_diagnostic | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | +0.000 |
| mentions_peak_time | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | +0.000 |

## Continuous metrics (mean ± std)

| metric | base | v2 | v3 |
|---|---:|---:|---:|
| yeo7_networks_mentioned | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| yeo7_any_alias | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| roi_verbatim_count | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| n_words | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| n_chars | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| ttr | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Paired deltas — v2 vs base (95% bootstrap CI + two-sided sign test)

| metric | mean ± std | 95% CI | sign-test p |
|---|---:|---:|---:|
| opens_with_template_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| has_tribe_disclaimer_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| has_not_diagnostic_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| mentions_peak_time_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_strict_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_alias_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| roi_verbatim_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| n_words_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| ttr_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |

## Paired deltas — v3 vs base (95% bootstrap CI + two-sided sign test)

| metric | mean ± std | 95% CI | sign-test p |
|---|---:|---:|---:|
| opens_with_template_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| has_tribe_disclaimer_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| has_not_diagnostic_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| mentions_peak_time_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_strict_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_alias_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| roi_verbatim_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| n_words_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| ttr_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |

## Paired deltas — v3 vs v2 (95% bootstrap CI + two-sided sign test)

| metric | mean ± std | 95% CI | sign-test p |
|---|---:|---:|---:|
| opens_with_template_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| has_tribe_disclaimer_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| has_not_diagnostic_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| mentions_peak_time_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_strict_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_alias_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| roi_verbatim_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| n_words_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| ttr_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
