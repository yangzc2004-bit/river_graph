# Expanded hydro-only upstream development results

| Model | MAE mg/L | Q90 MAE | log1p MAE |
|---|---:|---:|---:|
| available_real_integrated | 1.723002 | 8.895024 | 0.248207 |
| dynamic_lagged | 1.717226 | 8.852561 | 0.247090 |
| expanded_matched_nonupstream | 1.715839 | 8.852472 | 0.246881 |
| expanded_matched_upstream | 1.714690 | 8.843898 | 0.246699 |
| expanded_uniform | 1.713772 | 8.834304 | 0.246560 |
| expanded_upstream | 1.714729 | 8.841157 | 0.246694 |
| matched_state_nonupstream | 1.715176 | 8.856303 | 0.246832 |
| matched_state_upstream | 1.715799 | 8.853490 | 0.246827 |
| observed_state | 1.715815 | 8.848287 | 0.246815 |
| station_hidden_trees | 1.866362 | 9.394458 | 0.275922 |

| Candidate / reference | MAE reduction % [95% station CI] | Positive packages |
|---|---:|---:|
| expanded_upstream / available_real_integrated | 0.480 [0.082, 0.858] | 8/9 |
| expanded_uniform / available_real_integrated | 0.536 [0.149, 0.912] | 8/9 |
| expanded_matched_upstream / available_real_integrated | 0.482 [0.080, 0.867] | 8/9 |
| expanded_matched_nonupstream / available_real_integrated | 0.416 [-0.005, 0.828] | 8/9 |
| expanded_upstream / observed_state | 0.063 [-0.065, 0.204] | 4/9 |
| expanded_upstream / dynamic_lagged | 0.145 [0.016, 0.302] | 5/9 |
| expanded_upstream / station_hidden_trees | 8.125 [5.394, 11.210] | 9/9 |
| expanded_upstream / expanded_uniform | -0.056 [-0.108, -0.008] | 0/9 |
| expanded_matched_upstream / expanded_matched_nonupstream | 0.067 [-0.192, 0.299] | 5/9 |
| expanded_matched_upstream / matched_state_upstream | 0.065 [-0.048, 0.194] | 4/9 |
| expanded_matched_nonupstream / matched_state_nonupstream | -0.039 [-0.299, 0.218] | 2/9 |

Source-role development; complete and observed branch selection used these roles.
All fixed arms retained. Latent environmental states are distinct from measured DOC.
