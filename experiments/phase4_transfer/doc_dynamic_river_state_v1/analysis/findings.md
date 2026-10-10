# Upstream state development results

| Model | MAE mg/L | Q90 MAE | log1p MAE |
|---|---:|---:|---:|
| available_real_integrated | 1.723002 | 8.895024 | 0.248207 |
| dynamic_lagged | 1.717226 | 8.852561 | 0.247090 |
| matched_state_nonupstream | 1.715176 | 8.856303 | 0.246832 |
| matched_state_upstream | 1.715799 | 8.853490 | 0.246827 |
| observed_state | 1.715815 | 8.848287 | 0.246815 |
| state_upstream | 1.720945 | 8.895400 | 0.247863 |
| station_hidden_trees | 1.866362 | 9.394458 | 0.275922 |
| uniform_state | 1.718931 | 8.890391 | 0.247477 |

| Candidate / reference | MAE reduction % [95% station CI] | Positive packages |
|---|---:|---:|
| state_upstream / available_real_integrated | 0.119 [0.012, 0.260] | 2/9 |
| observed_state / available_real_integrated | 0.417 [-0.012, 0.816] | 8/9 |
| uniform_state / available_real_integrated | 0.236 [0.004, 0.509] | 5/9 |
| matched_state_upstream / available_real_integrated | 0.418 [0.007, 0.805] | 8/9 |
| matched_state_nonupstream / available_real_integrated | 0.454 [0.060, 0.845] | 8/9 |
| observed_state / dynamic_lagged | 0.082 [-0.046, 0.226] | 4/9 |
| observed_state / state_upstream | 0.298 [-0.115, 0.662] | 8/9 |
| observed_state / station_hidden_trees | 8.066 [5.352, 11.139] | 9/9 |
| state_upstream / uniform_state | -0.117 [-0.356, 0.099] | 2/9 |
| matched_state_upstream / matched_state_nonupstream | -0.036 [-0.199, 0.124] | 2/9 |
| matched_state_upstream / dynamic_lagged | 0.083 [-0.033, 0.218] | 5/9 |

Source-role development; complete and observed branch selection used these roles.
All fixed arms retained. Latent environmental states are distinct from measured DOC.
