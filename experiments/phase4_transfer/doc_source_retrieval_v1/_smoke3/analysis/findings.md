# DOC source-retrieval development

These results use selected source-validation stations only.
No target DOC evaluation or geographical/external confirmation has been performed.

## All-observation K0 MAE

| Model | MAE mg/L | Q90 MAE |
|---|---:|---:|
| current_model | 1.969191 | nan |
| hydro_pretrained | 2.137867 | nan |
| hydro_pretrained_retrieval | 2.137867 | nan |
| hydro_pretrained_retrieval_uniform | 2.137867 | nan |
| hydro_pretrained_retrieval_zero_source_values | 2.137867 | nan |
| matched_daily_trees | 2.312927 | nan |
| retrieval | 1.969191 | nan |
| retrieval_uniform | 1.969191 | nan |
| retrieval_zero_source_values | 1.969191 | nan |
| static_memory | 1.969191 | nan |

## Paired effects

| Candidate vs reference | MAE gain % [95% CI] |
|---|---:|
| retrieval vs current_model | 0.000 [0.000, 0.000] |
| retrieval vs matched_daily_trees | 14.862 [9.952, 20.735] |
| retrieval vs static_memory | 0.000 [0.000, 0.000] |
| hydro_pretrained vs current_model | -8.566 [-14.100, -3.534] |
| hydro_pretrained vs matched_daily_trees | 7.569 [3.824, 12.233] |
| hydro_pretrained vs static_memory | -8.566 [-14.100, -3.534] |
| hydro_pretrained_retrieval vs current_model | -8.566 [-14.100, -3.534] |
| hydro_pretrained_retrieval vs matched_daily_trees | 7.569 [3.824, 12.233] |
| hydro_pretrained_retrieval vs static_memory | -8.566 [-14.100, -3.534] |
| retrieval vs retrieval_uniform | 0.000 [0.000, 0.000] |
| retrieval vs retrieval_zero_source_values | 0.000 [0.000, 0.000] |
| hydro_pretrained_retrieval vs hydro_pretrained_retrieval_uniform | 0.000 [0.000, 0.000] |
| hydro_pretrained_retrieval vs hydro_pretrained_retrieval_zero_source_values | 0.000 [0.000, 0.000] |
| hydro_pretrained_retrieval vs retrieval | -8.566 [-14.100, -3.534] |

Intervals describe the selected development panel; selection optimism remains.
Geographical confirmation is required before spatial-performance claims.
