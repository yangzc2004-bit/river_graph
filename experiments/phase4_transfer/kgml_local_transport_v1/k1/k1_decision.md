# K1 decision: local correction is useful; river-message gain is not isolated

K1 completed the planned 36 runs: DOC, two missingness families, three seeds,
and six arms. The paired analysis uses the same hidden cells for every arm and
station-clustered bootstrap intervals.

## What the pilot establishes

The residual construction is useful in the temporal extrapolation setting. In
`e2a_strict`, the upstream residual arm reduces MAE from 1.205 for RF-local to
0.977 (21.9% point improvement; station-bootstrap 95% interval 0.152--0.377
mg/L). The no-message residual arm gives 0.976, essentially the same result.

In `e3_spatial_seed42`, the residual arms are close to RF-local (2.954 versus
2.971; about 0.2%), while RF-context is clearly better (2.679). The residual
construction therefore helps mainly in temporal extrapolation and does not
replace explicit spatial context in this pilot.

## What the pilot does not establish

The river message itself is not identified as the source of the improvement.
Upstream versus no-message changes MAE by about -0.005 mg/L in `e2a_strict`
(95% interval -0.011--0.001) and by less than 0.001 mg/L in the spatial mask.
The bidirectional arm is equally close to no-message. The learned graph delta
is also nearly a global correction (standard deviation about 0.01--0.02 mg/L
in `e2a_strict`, and below 0.001 mg/L in the spatial mask), rather than a
strongly varying station-specific transport signal.

The correct K1 interpretation is therefore:

> An OOF-trained residual correction improves the local RF baseline for DOC
> temporal extrapolation, but this pilot has not yet shown that directed river
> messages add information beyond the local temporal representation.

## Next experiment

Do not start a large lagged-transport matrix from this result. First run a
small source-isolation pilot in which the residual head receives an explicit
message-only representation and the no-message arm is structurally zero. Keep
the same two masks and three seeds. This separates a genuine upstream signal
from a learned global bias correction. If the message-only arm shows a stable
gain in `e2a_strict`, then proceed to fixed lag buckets; otherwise close the
river-message branch and retain the residual model as a conditional temporal
correction.

The complete numerical audit is in `paired_bootstrap_summary.csv`, with the
cell-level matched products in `paired_seedmean_cells.csv`.
