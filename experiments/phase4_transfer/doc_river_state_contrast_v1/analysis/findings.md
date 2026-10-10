# Upstream-to-local environmental contrast development results

| Model | MAE mg/L | Q90 MAE | log1p MAE |
|---|---:|---:|---:|
| available_real_integrated | 1.723002 | 8.895024 | 0.248207 |
| contrast_matched_nonupstream | 1.716690 | 8.854154 | 0.246987 |
| contrast_matched_upstream | 1.715867 | 8.852491 | 0.246824 |
| contrast_uniform | 1.715912 | 8.851760 | 0.246846 |
| contrast_upstream | 1.715889 | 8.853289 | 0.246856 |
| dynamic_lagged | 1.717226 | 8.852561 | 0.247090 |
| expanded_matched_nonupstream | 1.715839 | 8.852472 | 0.246881 |
| expanded_matched_upstream | 1.714690 | 8.843898 | 0.246699 |
| expanded_uniform | 1.713772 | 8.834304 | 0.246560 |
| expanded_upstream | 1.714729 | 8.841157 | 0.246694 |
| observed_state | 1.715815 | 8.848287 | 0.246815 |
| station_hidden_trees | 1.866362 | 9.394458 | 0.275922 |

| Candidate / reference | MAE reduction % [95% station CI] | Positive packages |
|---|---:|---:|
| contrast_upstream / available_real_integrated | 0.413 [0.020, 0.778] | 9/9 |
| contrast_uniform / available_real_integrated | 0.411 [0.025, 0.776] | 9/9 |
| contrast_matched_upstream / available_real_integrated | 0.414 [0.027, 0.776] | 9/9 |
| contrast_matched_nonupstream / available_real_integrated | 0.366 [-0.027, 0.748] | 8/9 |
| contrast_upstream / expanded_upstream | -0.068 [-0.213, 0.068] | 1/9 |
| contrast_upstream / dynamic_lagged | 0.078 [-0.009, 0.166] | 4/9 |
| contrast_upstream / station_hidden_trees | 8.062 [5.320, 11.161] | 9/9 |
| contrast_upstream / contrast_uniform | 0.001 [-0.042, 0.046] | 1/9 |
| contrast_matched_upstream / contrast_matched_nonupstream | 0.048 [-0.089, 0.168] | 4/9 |
| contrast_matched_upstream / expanded_matched_upstream | -0.069 [-0.234, 0.081] | 1/9 |
| contrast_matched_nonupstream / expanded_matched_nonupstream | -0.050 [-0.299, 0.158] | 1/9 |

Source-role development; complete and observed branch selection used these roles.
All fixed arms retained. Latent environmental states are distinct from measured DOC.
