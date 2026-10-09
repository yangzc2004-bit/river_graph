# Source monthly hydrology in experience selection

## Research question

The existing receiving GRU uses monthly temperature and discharge, but the
individual source attention keys contain ecology and eight daily-flow summaries
without source monthly temperature or discharge. Test whether selecting source
DOC residuals using their actual monthly water conditions improves zero-water-
quality station reconstruction. This extends the existing first-order full-
source-level model; it adds no backbone, output transform or forest fitting.

## Fixed source-development experiment

Use source training/validation roles142/143/144 and seeds42/43/44. The complete
source-level attention version is the immediate comparator. Carry its outputs,
original complete model and strong station-hidden trees unchanged. No previously
scored geographical or external DOC is used for this study's model selection.

Three learned-attention arms differ only in four appended source key columns:

1. Current source temperature, its visibility, discharge, its visibility.
2. Source station observed-period hydro means, with current visibility retained.
3. Four zeros, a matched-parameter input-removal control.

The source-mean control is computed from frozen source inputs over their training
calendar. It tests between-month source conditions in a spatial reconstruction
task; it is not an online causal hydrological climatology. All receiving queries,
source DOC values, candidates, daily keys, ecology, visibility and native heads
remain the same. Hidden hydro values are zeroed. The new four key coefficients
start at zero, preserving the earlier attention allocation at initialization.
The receiving query continues to use its existing GRU hydro history.

All arms retain two32-dimensional heads,12-month GRU,30epochs/patience5 and
the existing native-scale tail loss and validation-selected fusion. The change
adds256 coefficients to the37,900-parameter model.27neural fits,90 reused
double-held-fold references and no new forest fits. Source candidate and DOC
experience exclusion is unchanged: query fold absent from the source library;
both query and donor folds absent from that donor's reference. New-site K0
receives no DOC/pH/conductance or their derived historical/support inputs.

## Analysis and next step

Report source-validation MAE, Q90 error, signed bias, station-equal error,
partition/seed consistency and paired5,000 station bootstrap. Compare current
monthly keys against the immediate full-source model and both matched controls;
report native-only and complete fusion separately. Save allocation entropy in
raw Shannon units, prior mass and learned monthly-key coefficient changes.

First run142/42 and verify save/load, candidate/source-role equality, finite
products and future-input invisibility. Then complete all nine fixed packages.
Promote an informative stable increment to a separate fixed geographical study;
otherwise close this source-key variant and retain the preceding model. Never
select a different region/K winner or retune using retained external outcomes.
