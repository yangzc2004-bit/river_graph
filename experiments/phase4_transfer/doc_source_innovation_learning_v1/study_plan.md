# Learning time-aligned source information in the existing DOC residual

## Question

The scalar source-innovation probe improves source-development MAE by0.303%
[-0.107%,0.747%]. It uses contemporaneous source DOC while receiving stations
remain entirely water-quality-free. This is a small, uncertain development
signal. Test whether the retained ecology/GRU state can learn when to use it.
This plan follows those observed development results; old geographical and
external outcomes do not select the new mechanism.

## Existing model and one information change

Keep the retained log environmental reference, station-hidden inputs, native
residual target, ecological self encoder, observation GRU,12-month window,
initial weights,30-epoch/patience5 budget, learning rates and tail weight2.
Append three current-month features to the existing residual readout:

1. Source innovation in native mg/L, using a fixed1mg/L unit scale.
2. Matched source count divided by20.
3. Supported weight mass divided by1+mass.

Keep the four existing hidden-feature interactions and add the innovation
interaction with the SAME GRU hidden state. No new backbone, graph depth or
attention operator is added. The zero-initialized head has67 extra trainable
parameters. Missing source information has zero native innovation. The original
environmental and local-history input channels remain unchanged.

Three matched neural arms use identical shapes, masks and support:

- Real same-month innovation.
- Earlier-same-season innovation, with the causal control already tested.
- Availability only: innovation and its interaction are zero; support remains.

Refit two strong ExtraTrees arms with these same real/historical feature triples
and retained forest settings. The forest's target transform remains log1p.
The neural model retains the original forest/OOF reference, so tree information
and neural information can be evaluated separately. Fuse the fitted neural
prediction with the same static ecological memory procedure; compare complete
with complete and neural-only with neural-only.

## Source station exclusion

Use142/143/144 ×42/43/44. For training a query fold A, source-library labels
come only from the other four folds. For a donor fold B, its environmental
reference is fitted on neither A nor B. Fit each unordered pair A/B once;
ten pair-reference fits per package supply both directions. Hide A/B values
from target-derived training/inference features before fitting. Save the new
reference states, selected-cell predictions and fitted/hidden station roles.

The45 reference-trajectory forests were inspected and each omits only one
fold. Their geometry and frozen hyperparameters can be inspected, but their
fitted weights cannot be reused as double-fold OOF references. This is an
explicit correction to the earlier suggestion that existing fits might suffice.
Ninety new pair-reference fits are required across the nine packages.

Construct source seasonal means/innovations using these pair-held donor
references. Each query fold has its own excluded library and source-derived
sigma. At validation, the library uses the original source station-OOF reference,
with all source training stations and no validation/test station labels.
Use the same fixed unit scale for all three neural arms and both tree arms.
This choice is made before any new fit: a shared scale estimated across all
episode libraries could indirectly depend on a query fold through other banks.
The fixed unit scale preserves its full exclusion and comparable units.
Keep the same receiver query
and source-derived Q90 thresholds. Never select a different model by K/region.

## Execution and interpretation

Implement/test nested exclusion, future visibility, initial zero-head equality,
new-node inference and save/load. Fit/replay the first package, then all nine.
Use5,000 paired station draws, source-support and Q90/bias/station diagnostics,
and inspect source-backed figures. Do not restart previous geography/external
runners. Interpret real versus historical/availability arms as an information
comparison. Improvements over older models or trees are not improvements over
the retained complete model. Keep small or null effects in the research record.

If the learned version has useful, consistent source-development gains, proceed
to a separate fixed geographical confirmation version. Otherwise close this
mechanism without expanding neighbour/lag/architecture searches. The current
deployed release is retained until a new complete model is confirmed.
