# River structure development results

DOC reconstruction at receiving stations without water-quality inputs. These are source-validation
development results; geographical and external confirmation products remain unchanged.

Completed packages: 7. Partial report: True.

## Reconstruction error

| Model | MAE mg/L | Q90 MAE |
|---|---:|---:|
| available_real_integrated | 1.723036 | 8.870288 |
| available_real_native | 1.727496 | 8.800533 |
| river_none | 1.723063 | 8.870124 |
| river_none_zero_river | 1.723063 | 8.870124 |
| river_rewired | 1.721242 | 8.876645 |
| river_rewired_zero_river | 1.722638 | 8.880451 |
| river_simple | 1.722203 | 8.842522 |
| river_simple_zero_river | 1.727731 | 8.867327 |
| river_structure | 1.723027 | 8.862114 |
| river_structure_zero_river | 1.729060 | 8.887950 |
| station_hidden_trees | 1.863650 | 9.372972 |

## River comparisons

| Candidate vs reference | MAE reduction % and 95% interval |
|---|---:|
| river_none vs available_real_integrated | -0.002 [-0.007, 0.004] |
| river_none vs station_hidden_trees | 7.544 [4.793, 10.707] |
| river_simple vs available_real_integrated | 0.048 [-0.372, 0.450] |
| river_simple vs station_hidden_trees | 7.590 [4.843, 10.695] |
| river_structure vs available_real_integrated | 0.001 [-0.467, 0.455] |
| river_structure vs station_hidden_trees | 7.546 [4.923, 10.514] |
| river_rewired vs available_real_integrated | 0.104 [-0.105, 0.315] |
| river_rewired vs station_hidden_trees | 7.641 [4.917, 10.737] |
| river_simple vs river_none | 0.050 [-0.371, 0.453] |
| river_structure vs river_none | 0.002 [-0.464, 0.457] |
| river_rewired vs river_none | 0.106 [-0.102, 0.314] |
| river_structure vs river_simple | -0.048 [-0.316, 0.217] |
| river_structure vs river_rewired | -0.104 [-0.609, 0.373] |
| river_simple vs river_simple_zero_river | 0.320 [-0.007, 0.638] |
| river_structure vs river_structure_zero_river | 0.349 [0.037, 0.662] |
| river_rewired vs river_rewired_zero_river | 0.081 [-0.035, 0.216] |

The matched no-river arm separates joint retraining from added connectivity.
The fitted zero-river product isolates the direct message contribution within each fitted network.
Rewiring preserves candidate slots and matches source drainage area; actual observation availability
can differ and is reported. Attention weights are predictive allocation, not physical transport rates.
Structure profiles overlap and subgroup comparisons remain exploratory.
