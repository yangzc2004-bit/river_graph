# River structure development results

DOC reconstruction at receiving stations without water-quality inputs. These are source-validation
development results; geographical and external confirmation products remain unchanged.

Completed packages: 9. Partial report: False.

## Reconstruction error

| Model | MAE mg/L | Q90 MAE |
|---|---:|---:|
| available_real_integrated | 1.723002 | 8.895024 |
| available_real_native | 1.726375 | 8.840746 |
| river_none | 1.723029 | 8.894860 |
| river_none_zero_river | 1.723029 | 8.894860 |
| river_rewired | 1.722055 | 8.903378 |
| river_rewired_zero_river | 1.723076 | 8.902103 |
| river_simple | 1.723066 | 8.883750 |
| river_simple_zero_river | 1.728201 | 8.908923 |
| river_structure | 1.724014 | 8.879511 |
| river_structure_zero_river | 1.728806 | 8.907517 |
| station_hidden_trees | 1.866362 | 9.394458 |

## River comparisons

| Candidate vs reference | MAE reduction % and 95% interval |
|---|---:|
| river_none vs available_real_integrated | -0.002 [-0.007, 0.004] |
| river_none vs station_hidden_trees | 7.680 [4.958, 10.783] |
| river_simple vs available_real_integrated | -0.004 [-0.413, 0.388] |
| river_simple vs station_hidden_trees | 7.678 [5.007, 10.656] |
| river_structure vs available_real_integrated | -0.059 [-0.548, 0.405] |
| river_structure vs station_hidden_trees | 7.627 [5.051, 10.507] |
| river_rewired vs available_real_integrated | 0.055 [-0.124, 0.257] |
| river_rewired vs station_hidden_trees | 7.732 [5.040, 10.791] |
| river_simple vs river_none | -0.002 [-0.410, 0.390] |
| river_structure vs river_none | -0.057 [-0.544, 0.407] |
| river_rewired vs river_none | 0.057 [-0.121, 0.256] |
| river_structure vs river_simple | -0.055 [-0.342, 0.229] |
| river_structure vs river_rewired | -0.114 [-0.634, 0.368] |
| river_simple vs river_simple_zero_river | 0.297 [0.003, 0.591] |
| river_structure vs river_structure_zero_river | 0.277 [-0.037, 0.577] |
| river_rewired vs river_rewired_zero_river | 0.059 [-0.041, 0.183] |

The matched no-river arm separates joint retraining from added connectivity.
The fitted zero-river product isolates the direct message contribution within each fitted network.
Rewiring preserves candidate slots and matches source drainage area; actual observation availability
can differ and is reported. Attention weights are predictive allocation, not physical transport rates.
Structure profiles overlap and subgroup comparisons remain exploratory.
