# v2 brain-LoRA eval statistics (n=30 paired samples)

## Style-transfer rates (binary)

| metric | base | adapter | delta |
|---|---:|---:|---:|
| opens_with_template | 0.000 | 0.733 | +0.733 |
| has_tribe_disclaimer | 1.000 | 1.000 | +0.000 |
| has_not_diagnostic | 0.133 | 1.000 | +0.867 |
| mentions_peak_time | 0.667 | 0.967 | +0.300 |

## Continuous metrics (mean ± std)

| metric | base | adapter | mean delta |
|---|---:|---:|---:|
| yeo7_networks_mentioned | 3.133 ± 1.697 | 0.400 ± 0.894 | -2.733 |
| yeo7_any_alias | 4.300 ± 0.794 | 4.000 ± 0.788 | -0.300 |
| roi_verbatim_count | 0.733 ± 1.982 | 0.333 ± 1.493 | -0.400 |
| n_words | 105.433 ± 9.954 | 118.233 ± 7.704 | +12.800 |
| n_chars | 814.833 ± 62.579 | 923.400 ± 69.697 | +108.567 |
| ttr | 0.764 ± 0.032 | 0.714 ± 0.036 | -0.050 |

## Paired within-sample deltas (adapter - base), 95% bootstrap CI + sign test

| metric | mean ± std | 95% CI | sign-test p (two-sided) |
|---|---:|---:|---:|
| opens_with_template_delta | +0.733 ± 0.450 | [+0.567, +0.867] | 4.77e-07 |
| has_tribe_disclaimer_delta | +0.000 ± 0.000 | [+0.000, +0.000] | 1 |
| yeo7_strict_delta | -2.733 ± 2.227 | [-3.500, -1.933] | 4.17e-07 |
| yeo7_alias_delta | -0.300 ± 0.535 | [-0.500, -0.100] | 0.0117 |
| roi_verbatim_delta | -0.400 ± 2.527 | [-1.300, +0.500] | 0.375 |
| n_words_delta | +12.800 ± 11.990 | [+8.667, +17.100] | 1.52e-05 |
| ttr_delta | -0.050 ± 0.050 | [-0.068, -0.033] | 0.000325 |
