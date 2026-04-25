# v2 brain-LoRA eval statistics (n=8 paired samples)

## Style-transfer rates (binary)

| metric | base | adapter | delta |
|---|---:|---:|---:|
| opens_with_template | 0.000 | 0.875 | +0.875 |
| has_tribe_disclaimer | 0.000 | 1.000 | +1.000 |
| has_not_diagnostic | 0.000 | 1.000 | +1.000 |
| mentions_peak_time | 0.500 | 1.000 | +0.500 |

## Continuous metrics (mean ± std)

| metric | base | adapter | mean delta |
|---|---:|---:|---:|
| yeo7_networks_mentioned | 3.375 ± 1.408 | 0.625 ± 1.408 | -2.750 |
| yeo7_any_alias | 4.375 ± 1.188 | 3.875 ± 0.641 | -0.500 |
| roi_verbatim_count | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 |
| n_words | 98.250 ± 6.018 | 122.375 ± 10.743 | +24.125 |
| n_chars | 765.125 ± 46.001 | 930.000 ± 71.073 | +164.875 |
| ttr | 0.756 ± 0.037 | 0.694 ± 0.032 | -0.062 |

## Paired within-sample deltas (adapter - base), 95% bootstrap CI + sign test

| metric | mean ± std | 95% CI | sign-test p (two-sided) |
|---|---:|---:|---:|
| opens_with_template_delta | +0.875 ± 0.354 | [+0.625, +1.000] | 0.0156 |
| has_tribe_disclaimer_delta | +1.000 ± 0.000 | [+1.000, +1.000] | 0.00781 |
| yeo7_strict_delta | -2.750 ± 1.909 | [-4.000, -1.500] | 0.0156 |
| yeo7_alias_delta | -0.500 ± 0.756 | [-1.000, -0.125] | 0.25 |
| roi_verbatim_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| n_words_delta | +24.125 ± 10.602 | [+17.250, +31.500] | 0.00781 |
| ttr_delta | -0.062 ± 0.060 | [-0.100, -0.024] | 0.0703 |
