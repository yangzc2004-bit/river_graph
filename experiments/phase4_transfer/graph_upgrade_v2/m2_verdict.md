# M2 mechanism result

M2 is complete: 18 runs using the M1 observation features plus directed
upstream messages in the `{0, 1, 3, 6, 12}` month lag buckets.

| Analyte | Temporal holdout MAE | Spatial holdout MAE | M1 temporal | M1 spatial |
|---|---:|---:|---:|---:|
| DOC | 1.364 | 3.791 | 1.142 | 3.490 |
| pH | 0.261 | 0.293 | 0.272 | 0.280 |
| Specific conductance | 210.82 | 527.69 | 190.97 | 489.47 |

Lagged transport still improves over the released H2X-T temporal baseline in
all six analyte/family combinations, but it does not improve over M1 except
for the pH temporal holdout. The strongest M1 gains therefore come from
explicit observation support and age, not from adding a lagged message path.
The lagged model also has wider seed variation for conductance, which is
consistent with a less stable transport correction.

M2 answers the second mechanism question for this dataset: a physically
plausible lag parameterization is not enough to make river messages broadly
useful. The next experiment is M3, which keeps the observation-aware inputs
and tests whether the temporal representation, rather than message timing,
is the remaining limitation.

