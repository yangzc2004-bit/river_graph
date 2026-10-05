# DOC source-retrieval development

These results use selected source-validation stations only.
No target DOC evaluation or geographical/external confirmation has been performed.

## All-observation K0 MAE

| Model | MAE mg/L | Q90 MAE |
|---|---:|---:|
| current_model | 1.829579 | 9.225863 |
| hydro_pretrained | 1.835878 | 9.242435 |
| hydro_pretrained_retrieval | 1.835878 | 9.242435 |
| hydro_pretrained_retrieval_uniform | 1.835878 | 9.242435 |
| hydro_pretrained_retrieval_zero_source_values | 1.835878 | 9.242435 |
| matched_daily_trees | 1.963925 | 9.618575 |
| retrieval | 1.829579 | 9.225863 |
| retrieval_uniform | 1.829579 | 9.225863 |
| retrieval_zero_source_values | 1.829579 | 9.225863 |
| static_memory | 1.829579 | 9.225863 |

## Paired effects

| Candidate vs reference | MAE gain % [95% CI] |
|---|---:|
| retrieval vs current_model | 0.000 [0.000, 0.000] |
| retrieval vs matched_daily_trees | 6.841 [3.503, 10.350] |
| retrieval vs static_memory | 0.000 [0.000, 0.000] |
| hydro_pretrained vs current_model | -0.344 [-1.553, 0.770] |
| hydro_pretrained vs matched_daily_trees | 6.520 [3.412, 9.687] |
| hydro_pretrained vs static_memory | -0.344 [-1.553, 0.770] |
| hydro_pretrained_retrieval vs current_model | -0.344 [-1.553, 0.770] |
| hydro_pretrained_retrieval vs matched_daily_trees | 6.520 [3.412, 9.687] |
| hydro_pretrained_retrieval vs static_memory | -0.344 [-1.553, 0.770] |
| retrieval vs retrieval_uniform | 0.000 [0.000, 0.000] |
| retrieval vs retrieval_zero_source_values | 0.000 [0.000, 0.000] |
| hydro_pretrained_retrieval vs hydro_pretrained_retrieval_uniform | 0.000 [0.000, 0.000] |
| hydro_pretrained_retrieval vs hydro_pretrained_retrieval_zero_source_values | 0.000 [0.000, 0.000] |
| hydro_pretrained_retrieval vs retrieval | -0.344 [-1.553, 0.770] |

Intervals describe the selected development panel; selection optimism remains.
Geographical confirmation is required before spatial-performance claims.
