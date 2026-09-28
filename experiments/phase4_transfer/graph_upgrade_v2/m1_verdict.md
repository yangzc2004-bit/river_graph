# M1 mechanism result

The M1 pilot is complete: three analytes, two missingness families and three
seeds (18 runs, 30 training epochs). The observation-aware model adds recent
target support, observation age and a causal recurrent decay to the existing
H2X-T trunk.

## Mean test MAE

| Analyte | Temporal holdout (`e2a_strict`) | Spatial holdout (`e3_spatial_seed42`) | H2X-T temporal | H2X-T spatial |
|---|---:|---:|---:|---:|
| DOC | 1.142 | 3.490 | 1.223 | 3.032 |
| pH | 0.272 | 0.280 | 0.302* | 0.284* |
| Specific conductance | 190.97 | 489.47 | 221.76 | 448.62 |

\* The existing pH H2X-T archive has only the original 10-epoch matched
budget; DOC and conductance use the 30-epoch convergence archive.

The improvement is therefore conditional on both analyte and missingness
family. Relative to the matched 30-epoch baseline, DOC improves 6.6% and
conductance improves 13.9% in the temporal holdout, while both worsen in the
spatial holdout. pH improves against its available 10-epoch archive in the
temporal holdout but is essentially unchanged in the spatial holdout. The
spatial DOC predictions remain well behind the temporal random forest (MAE
2.683 in the matched RF runs), so M1 is
an observation-memory improvement, not a complete replacement for the
feature-based baseline.

The same pattern appears in the high-value tail: M1's Q90 MAE is 7.16 versus
11.05 for spatial DOC and 946 versus 1,921 for spatial specific conductance.
Tail errors remain large in absolute units, so the gain reflects a reduction in
extreme-error exposure rather than accurate reconstruction of every high-value
event.

## Scientific interpretation

M1 supports the first mechanism hypothesis: the age and support of a target
observation contain useful information beyond the raw target-value channel.
Its spatial-holdout behavior shows that this information does not solve
spatial transfer by itself. The result makes lagged transport the next useful
experiment: keep the observation memory, then test whether upstream support
becomes more informative when messages are allowed to arrive at realistic
monthly lags.

M2 is consequently started with the same analytes, masks, seeds and 30-epoch
budget. The comparison remains against H2X-T and temporal RF, and the M1
results remain a completed mechanism result rather than a tuning target.
