# Bounded chemical-state interpolation: source-validation pilot

## Question and scope

The chemical-coordinate calibration improved held-station K3 reconstruction,
but K5 improvement was smaller. Source-only diagnostics found useful station
mean corrections and weaker within-station shape corrections. This pilot tests
whether the remaining support residual can be transferred with a small nonlinear
interpolator instead of adding another prediction network.

Keep the completed chemical-support experiment, selected representations,
linear calibrators, ecological mixtures, neural heads and forests fixed. DOC
remains the sole target. Use all nine existing development packages, three
pipelines (direct neural, integrated neural and chemical tree), and the same
source-validation support/query schedule. No target outcomes are evaluated
and no neural network or forest is fitted in this pilot.

## Operator

For each station, reproduce the complete selected linear log-residual correction
using all K support readings. Evaluate that correction at support dates before
output clipping, and subtract it from each support's original log residual.
The kernel uses the centered remainder among chemically active supports.

Use the frozen two-dimensional source chemical coordinates. Normalize RBF
weights over active support dates; the weighted centered remainder is an additive
log-space correction. Its magnitude is bounded by the centered donor residual
range, multiplied by eta. No active query, fewer than two active supports,
identical support coordinates or eta0 gives exactzero. The existing final
no-chemistry fallback remains unchanged.

For each coordinate mode (chemistry or availability-only), freeze its bandwidth
unit to the median positive within-source-station coordinate distance. Use at
most24 deterministic, evenly spaced active months per source station. Only
source chemical features fit this scale; DOC labels and held-station coordinates
do not fit it. A degenerate distance inventory uses scale1 and is recorded.

Tune eta in{0,.25,.5,1} and bandwidth multipliers{.5,1,2} on active validation
query MAE, separately for pipeline, mode and K3/K5. Ties prefer eta0/smaller eta,
then multiplier1, then the smaller multiplier. The availability-only kernel is
a matched information control for the added kernel; the frozen parent and its
selected linear basis can already use chemical values. K0/K1 retain the parent
exactly.

Also evaluate incremental parameter selection with five validation-station
folds. Sort validation station IDs, permute with RNG seed3100+partition seed,
and split into five groups using `numpy.array_split`. Use identical folds
across training seeds, pipelines and coordinate modes. Choose eta and bandwidth
on the other four folds' active queries, and score the held fold; save station
loss totals for all candidates so these choices are independently reproducible.
This evaluates the added kernel's parameter transfer conditional on a frozen
parent already selected using validation data. It does not make the parent an
OOF predictor or substitute for later fresh-partition confirmation.

## Products and decision

Save source-distance definitions, fixed linear remainder checks, all validation
candidate scores, selected parameters, validation-only predictions, station
responses and whole K curves. Independently reconstruct weights, remainders
and selection. Report both coordinate modes and all pipelines. Selection scores
are calibration development evidence, not unbiased confirmation.

If the integrated neural interpolation provides coherent additional improvement
under incremental station-fold validation, freeze it for a later fresh-partition comparison with the retained
chemical-coordinate model and chemical tree. If it does not, retain the simpler
completed chemical-coordinate model and move to fresh-partition confirmation.
This is a bounded support-transfer test; chemical distance is not assumed to
measure physical transport or distance-dependent reliability.
