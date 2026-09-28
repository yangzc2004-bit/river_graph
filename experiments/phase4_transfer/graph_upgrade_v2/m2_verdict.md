# M2 mechanism result

M2 is complete: 18 runs using the M1 observation features plus directed
upstream messages in the `{0, 1, 3, 6, 12}` month lag buckets.

| Analyte | Temporal holdout MAE | Spatial holdout MAE | M1 temporal | M1 spatial |
|---|---:|---:|---:|---:|
| DOC | 1.364 | 3.791 | 1.223 | 3.032 |
| pH | 0.261 | 0.293 | 0.302* | 0.284* |
| Specific conductance | 210.82 | 527.69 | 221.76 | 448.62 |

\* The existing pH H2X-T archive has only the original 10-epoch matched
budget; DOC and conductance use the 30-epoch convergence archive.

Lagged transport improves over M1 only for pH in the temporal holdout. It is
better than the matched 30-epoch H2X-T baseline for temporal conductance but
does not reach M1, and is worse than that baseline for DOC and spatial
conductance. The strongest M1 gains therefore come from
explicit observation support and age, not from adding a lagged message path.
The lagged model also has wider seed variation for conductance, which is
consistent with a less stable transport correction.

M2 answers the second mechanism question for this dataset: a physically
plausible lag parameterization is not enough to make river messages broadly
useful. The next experiment is M3, which keeps the observation-aware inputs
and tests whether the temporal representation, rather than message timing,
is the remaining limitation.
