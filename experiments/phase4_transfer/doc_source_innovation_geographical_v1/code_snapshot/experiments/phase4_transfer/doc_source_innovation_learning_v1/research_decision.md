# Research decision: learning contemporaneous source DOC information

## Completed study

All nine source-development packages (partitions142/143/144, seeds42/43/44)
are complete:90 double-fold environmental references,27 neural residual fits
and18 enriched ExtraTrees fits. Receiving stations provide no DOC, pH or
conductance. The retained ecological self encoder, observation-aware GRU,
12-month history, native residual target and static ecological memory remain.
Three current readout inputs and one hidden-state interaction add67 parameters.
The training budget remains30 epochs with patience5.

The source-library reference excludes both the query and donor folds. Libraries
exclude the query fold's observations. Existing single-fold forests could not
supply this exclusion; the90 pair-reference fits were necessary. Validation
libraries contain source training observations only. Current source departures
are deviations from a fixed source seasonal residual climatology. This is a
spatial reconstruction study: permitted source training statistics use their
training record; it is not a temporal holdout protocol.

## Performance

Estimates equally weight the three partitions after averaging training seeds.
Intervals use5,000 paired station draws. Validation selected checkpoints and
fusion; these are development estimates, not independent confirmation.

| Complete procedure | MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete model |1.769053|9.075420|
| Same-month source information |1.761078|8.999148|
| Earlier-same-season source information |1.768392|9.066046|
| Source availability only |1.769076|9.064607|
| Strong trees |1.866362|9.394458|
| Trees with the same current source information |1.865680|9.396532|

Same-month complete improvement over the actual retained complete model is
**0.451% [-0.045%,1.032%]**. Two of three partition means and seven of nine
packages improve. Q90 improvement is **0.840% [0.150%,1.906%]**, also positive
in two partitions and seven packages. All three seed-average directions improve.
Partition143 is essentially unchanged in overall MAE; Q90 in partition142 is
worse. These outcomes remain in the main tables.

The same-month complete arm improves over the historical and availability
controls by0.414% and0.452%, respectively. Their overall intervals still cross
zero, although partition-average directions are positive in all three. Matched
neural-only improvement over retained neural-only is0.509% [-0.026%,1.122%];
its Q90 gain is0.850% [0.111%,1.967%]. Thus the complete-procedure result is
not merely a comparison with an older, weaker model.

The enriched tree's gain over the bare strong tree is0.037%
[-0.614%,0.690%], with a slightly worse Q90 point estimate. The5.607% advantage
of the same-month complete neural procedure over the equally informed tree
includes the retained model's existing advantage; it is not the isolated effect
of adding source information.

## What the information comparison supports

Earlier-season and availability controls preserve the same donor counts and
weights. The same-month arm has the best overall point estimate, with positive
control contrasts across the three development partitions. The high-DOC signal
is more resolved than the overall signal. This supports a limited fixed-version
geographical replication, without an architecture or neighbour-count search.

Support-stratum means improve both with and without matched current donors.
The no-donor group improves from2.004480 to1.993118mg/L. Learning changes the
local readout as well as its use of source information, so the full gain cannot
be assigned exclusively to contemporaneous donors. These strata are descriptive
diagnostics, not causal attribution or individually confirmed effects.

## Verification and repair record

All nine libraries, feature populations, neural predictions and memory fusions
replay bitwise. Forest reduction replay uses1e-12 tolerance. The original
analyzer rejected an equivalent tuple/list config representation; its original
source and execution snapshot remain. The separate R1 analyzer compares
canonical configs. Fitting, predictions and endpoints were unchanged.
`design_clarification.md` records that all seven retained interactions, rather
than the four incorrectly named in plan prose, were preserved from execution
start. Model dimension38→41 and31559→31626 trainable parameters are verified.

The full suite passed979 tests with two explicit skips; Ruff and the historical
artifact audit passed. The figure was actually inspected. Its original layout
is retained; moving the legend repaired a label overlap without changing data.

## Next research action

Carry the fixed same-month neural structure and matched enriched-tree control
to `doc_source_innovation_geographical_v1`, on the five established HUC4 roles.
First complete seeds42/43/44, then the same candidate and controls at45/46.
Retain whole-region K0, fixed-query K curves, Q90, station-equal and support/
ecology/hydro diagnostics. The candidate is fixed by source development;
geographical results do not choose different models for regions or K values.
These already evaluated ST357 roles constitute retrospective geographical
replication, not independent external validation. Keep the portable release,
previous external products and main manuscript unchanged until this result.
