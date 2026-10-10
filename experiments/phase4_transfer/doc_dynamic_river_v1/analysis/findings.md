# Dynamic river development results

## Complete fixed comparison

| Model | MAE mg/L | Q90 MAE | log1p MAE |
|---|---:|---:|---:|
| available_real_integrated | 1.723002 | 8.895024 | 0.248207 |
| dynamic_lagged | 1.717226 | 8.852561 | 0.247090 |
| dynamic_same_month | 1.718173 | 8.854349 | 0.247182 |
| matched_nonupstream | 1.721678 | 8.891393 | 0.247985 |
| matched_upstream | 1.717396 | 8.860705 | 0.247221 |
| static_upstream | 1.718552 | 8.862655 | 0.247283 |
| station_hidden_trees | 1.866362 | 9.394458 | 0.275922 |

## Paired native-MAE contrasts

| Candidate / reference | Gain % [95% station CI] | Positive packages |
|---|---:|---:|
| static_upstream / available_real_integrated | 0.258 [-0.015, 0.520] | 9/9 |
| dynamic_same_month / available_real_integrated | 0.280 [-0.090, 0.620] | 9/9 |
| dynamic_lagged / available_real_integrated | 0.335 [-0.039, 0.679] | 8/9 |
| matched_upstream / available_real_integrated | 0.325 [0.017, 0.646] | 8/9 |
| matched_nonupstream / available_real_integrated | 0.077 [-0.024, 0.213] | 5/9 |
| dynamic_lagged / static_upstream | 0.077 [-0.093, 0.229] | 6/9 |
| dynamic_lagged / dynamic_same_month | 0.055 [-0.035, 0.147] | 7/9 |
| dynamic_lagged / station_hidden_trees | 7.991 [5.252, 11.080] | 9/9 |
| matched_upstream / matched_nonupstream | 0.249 [-0.030, 0.533] | 7/9 |
| matched_upstream / dynamic_lagged | -0.010 [-0.153, 0.145] | 3/9 |

Development results on retained ST357 source roles; not independent confirmation.
All five fitted arms are retained, including zero-selected and negative results.
