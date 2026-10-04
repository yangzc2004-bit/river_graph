# Next model iteration: carry chemical information into station calibration

## Evidence and question

The chemistry decoder improves K0 held-station MAE by2.14%, including ordinary
and high DOC. At K5 that advantage contracts and the chemistry tree remains
better. The recurrent/ecological support basis predates this chemical input.
The next hypothesis is that station calibration needs to distinguish chemical
states when transferring a handful of DOC residual observations across months.
This is a hypothesis from development results, not a demonstrated failure cause.

## Focused experiment

Keep the fitted nonlinear decoder, source forest/base, recurrent/ecological
trunk, existing queries and nested support observations fixed. Compare its
existing two-dimensional GRU support basis with an augmented basis containing
two additional chemical coordinates. Derive those chemical coordinates from
the decoder's eight-dimensional embedding using source-only dimension reduction;
do not fit a projection or normalization on target-query DOC labels.

Use the same ridge and shrinkage candidates in the existing support adapter,
selected on held-source-validation station episodes. Apply the augmented
calibration to the nonlinear chemical model and the chemical tree, preserving
the former basis as a matched reference for each. Keep all K curves and
absent-chemistry parent fallback. K0 predictions must remain exactly unchanged.

Report whether the new basis reduces K3/K5 error, whether chemical state
similarity explains support-to-query gains, and whether high-DOC bias/false-high
tradeoffs improve. Preserve the old calibration if source validation rejects
the new representation. Do not splice target-selected winners across K.

Before fitting any new full backbone, this comparison can reuse the existing
trained states. A promising result should then receive confirmation on fresh
station partitions; repeated testing on the present development partitions is
not independent confirmation.
