# Conditional station-held readout training: complete research decision

## Question and answer

Does the current upstream-state correction improve when its training errors
come from a scalar readout that did not fit the receiving station? **Not in
this controlled experiment.** The station-held variant essentially ties the
previous upstream-state model and does not beat an identically refitted
in-sample control. Keep the preceding model; do not adopt this branch or
choose its uniform/matched arm retrospectively as the new primary model.

The experiment changes the river branch's TRAINING anchor while retaining
the complete inference anchor, encoder/GRU, source attention, ecological
coefficients, observed-DOC correction, environmental-state banks and queries.
Only the newly fitted scalar readout's loss excludes the complete receiving
fold. This is conditional readout cross-fitting, **not whole-model station OOF**:
the retained representation, donor inputs, queries and other fitted components
still contain historical source-supervision information from those folds.

## Completed experiment

Nine packages: ST357 development partitions 142/143/144 x seeds 42/43/44.
54 scalar-readout fits (all-source plus five station-held fits per package),
45 river-state fits (five arms per package). No new forest fits.
All new readouts use fixed 30 epochs with no held-label/validation epoch
selection. River branches use the preceding 30-epoch / patience-5 budget and
validation MAE including epoch0. Receiving K0 sites have no water-quality
inputs. This is source-validation development, not geographical confirmation
or external-basin validation.

v1 completed only partition142 before a source-anchor replay assertion caught
the missing local nonnegative clipping before memory blending. It remains
archived. v2 implements the exact two-stage combination and reruns all nine
packages. No arm, budget or endpoint was changed based on v1 scores.

## Prediction results

Seed-mean losses within partition, then three partitions equally weighted.

| Model | DOC MAE (mg/L) | Q90 MAE (mg/L) | log1p MAE |
|---|---:|---:|---:|
| Complete retained predictor |1.723002|8.895024|0.248207|
| Observed-DOC river anchor |1.717226|8.852561|0.247090|
| Previous absolute upstream-state branch |1.714729|8.841157|0.246694|
| Refitted in-sample readout training control |1.713788|8.850744|0.246580|
| Station-held readout training (primary) |1.714736|8.848964|0.246685|
| Station-held, uniform allocation |1.713318|8.820629|0.246447|
| Station-held, matched real upstream |1.714430|8.847238|0.246640|
| Station-held, matched non-upstream |1.716336|8.849108|0.246982|
| Strong matched trees |1.866362|9.394458|0.275922|

Five-thousand paired whole-station bootstrap draws; a station recurring across
partitions receives the same multiplicity. Seeds are averaged, not counted as
independent ecological samples. Native overall panel: 140 unique stations,
7,897 unique station-months (8,857 partition-cell occurrences); Q90: 91 stations,
774 unique cells. The main comparisons need no empty-partition redraws.

| Comparison | MAE reduction % [95% station interval] | Direction |
|---|---:|---|
| Station-held versus fitted-readout control |-0.0553[-0.1752,0.0410]|2/9 packages;1/3 partitions improve|
| Station-held versus previous upstream-state branch |-0.0004[-0.1375,0.1385]|2/9 packages;1/3 partitions improve|
| Station-held versus complete retained predictor |0.4798[0.1016,0.8524]|Accumulated development gain|
| Station-held versus observed-DOC anchor |0.1450[-0.0147,0.3305]|Additional state contribution unresolved|
| Learned versus uniform allocation |-0.0828[-0.2045,0.0570]|No established dynamic-allocation advantage|
| Matched real versus non-upstream |0.1111[-0.1481,0.3500]|No established independent connectivity advantage|

Q90 error rises 0.0883% [0.0271, 0.1893] against the previous upstream-state
branch. Against the refitted control, Q90 improves only 0.0201%
[-0.0318,0.0812]. Log-error intervals span zero for both incremental comparisons.
Station-equal gains are 0.0525% versus the preceding model and 0.0182% versus
the refit control: weighting changes the tiny point estimates but does not
yield a persuasive improvement. State coverage is exactly the previous bank:
31.8879%,36.1473%,64.3468% in the three partitions (44.1273% equally weighted).

Primary checkpoint epoch0 is selected 4/9 times; fitted control 5/9, uniform 4/9,
matched upstream 4/9 and matched non-upstream 7/9. Keep these zero-correction
outcomes in all tables.

## What the training-anchor diagnostic establishes

Conditional station-held readouts produce source MAE 1.625041 versus 1.573251
for the refitted in-sample readout (original fitted anchor 1.583507). Their
training errors are genuinely different. Mean log-residual Wasserstein distance
to the receiving-validation anchor changes 0.044904 to 0.042600. It decreases in
partitions 142/143 but increases in 144. This descriptive measure compares
different station populations; it is not a causal explanation of performance.

The original source-Q90 error already exceeds validation in 142/143, while 144
has validation-Q90 error 14.025 versus source 6.196 mg/L. Therefore the claim
"training errors are always too small" is not supported. More difficult
conditional training targets alone did not improve the river branch. This
does not determine the outcome of a genuinely fold-isolated complete predictor.

## Research decision and next experiment

Stop additional variants of this conditional scalar-readout adjustment. A
complete-model nested station refit remains a distinct, larger experiment;
the present result neither proves nor eliminates that possibility.

The next priority is an **explicit confluence and storage information operator**
on the retained river branch: mix incoming source departures with measured-flow
support (drainage area only as a separately labelled fallback), and use mapped
junction/storage attributes to regulate attenuation and temporal persistence.
These attributes already exist in the path features; the change should be how
messages combine, not more duplicate feature columns. Compare identical real
paths with an operator lacking the confluence/storage modulation, and retain
the matched non-ancestor control. Keep frozen local predictions, causal source
observations and zero-message fallback. Monthly source departures are learned
prediction corrections, not concentrations subject to exact mass conservation.
No simulated peak/buffering result is promoted to an empirical DOC conclusion.

## Verification and artifacts

All nine frozen readout designs rebuilt exactly. 54 readout-head predictions and
45 river checkpoints replayed; all retained model products remain bitwise
identical. Per-package first-fold label perturbation reproduces the readout
weights exactly conditional on the fixed design; it does not claim full-model
label isolation. Missing upstream-state predictions equal the original anchor.
All model MAE/log/Q90 metrics and 12 primary paired intervals independently
recalculated. Two PNG/PDF/SVG figures inspected; source/validation plots explicitly
describe different populations and expanded prediction-error axes.

Full suite: 1463 passed, 3 skipped, 8 warnings. Ruff passes; historical artifact
audit exits0. Local large fitting arrays and weights remain available but are
not committed. The verbose per-cell fitting-role caches also remain local;
compact station-fold summaries are in `analysis/readout_folds.json`.
Related source, tests, configuration, compact roles, predictions,
diagnostics, analysis and execution snapshots are committed together. No
training process remains running; no automation was started or resumed.

Reproduction: `uv run python scripts/run_ladder.py --experiment
doc-river-readout-crossfit-v2`; then the corresponding verify (with
`--rebuild-inputs`), analyze (`--bootstrap-draws 5000`), independent analysis
verification and plot scripts. Existing execution snapshots protect reruns
from changes in the working implementation; restoration of large arrays/weights
is required before prediction-only replay on another machine.
