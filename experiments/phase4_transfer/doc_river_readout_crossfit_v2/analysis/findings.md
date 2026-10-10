# Conditional station-held readout development

| Model | MAE mg/L | Q90 MAE | log1p MAE |
|---|---:|---:|---:|
| available_real_integrated | 1.723002 | 8.895024 | 0.248207 |
| crossfit_matched_nonupstream | 1.716336 | 8.849108 | 0.246982 |
| crossfit_matched_upstream | 1.714430 | 8.847238 | 0.246640 |
| crossfit_state | 1.714736 | 8.848964 | 0.246685 |
| crossfit_uniform | 1.713318 | 8.820629 | 0.246447 |
| dynamic_lagged | 1.717226 | 8.852561 | 0.247090 |
| expanded_matched_nonupstream | 1.715839 | 8.852472 | 0.246881 |
| expanded_matched_upstream | 1.714690 | 8.843898 | 0.246699 |
| expanded_uniform | 1.713772 | 8.834304 | 0.246560 |
| expanded_upstream | 1.714729 | 8.841157 | 0.246694 |
| refitted_state | 1.713788 | 8.850744 | 0.246580 |
| station_hidden_trees | 1.866362 | 9.394458 | 0.275922 |

| Candidate / reference | MAE reduction % [95% station CI] | Positive packages |
|---|---:|---:|
| crossfit_state / refitted_state | -0.055 [-0.175, 0.041] | 2/9 |
| crossfit_state / expanded_upstream | -0.000 [-0.137, 0.138] | 2/9 |
| crossfit_state / available_real_integrated | 0.480 [0.102, 0.852] | 8/9 |
| crossfit_state / dynamic_lagged | 0.145 [-0.015, 0.330] | 5/9 |
| crossfit_state / station_hidden_trees | 8.124 [5.396, 11.214] | 9/9 |
| refitted_state / expanded_upstream | 0.055 [-0.042, 0.175] | 3/9 |
| crossfit_state / crossfit_uniform | -0.083 [-0.205, 0.057] | 2/9 |
| crossfit_matched_upstream / crossfit_matched_nonupstream | 0.111 [-0.148, 0.350] | 5/9 |

Only newly fitted scalar-readout losses are station-held. All retained representations and river inputs remain conditional source fits.
Identical final inference anchor and state banks. This is development, not geographic/external confirmation.
