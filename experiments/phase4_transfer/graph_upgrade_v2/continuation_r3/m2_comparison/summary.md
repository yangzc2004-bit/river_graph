# M2 lag and message controls

All four M2 arms are complete: learned lag, fixed lag, same-month static
message, and no-message. Each arm has three analytes, two holdout families and
three seeds. The paired comparison uses the same hidden station-month cells
for every arm and reports station-cluster intervals.

## Main result

Learned and fixed lag are effectively indistinguishable. Their paired MAE
difference is below 0.03% for every analyte and holdout family. The learned
gate therefore did not discover a useful lag pattern beyond the fixed
available-lag weighting.

The static same-month arm is also close to the learned arm. Relative to the
no-message arm, message passing helps DOC temporal holdout by about 5.5%, but
the spatial DOC comparison is inconclusive. For pH, no-message is better on
the temporal holdout and the message arms are slightly better on the spatial
holdout. Conductance differences are small relative to station uncertainty.

This separates the mechanism question: the current data do not support a
general claim that learned river-message timing improves reconstruction. The
remaining M3 pilot tests whether a richer temporal representation can improve
the observation-aware trunk without adding another graph message mechanism.

The numerical tables are in `paired_summary.csv` and
`paired_seed_metrics.csv`; the run inventory records the completed products.
