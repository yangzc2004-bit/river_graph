# Conditional-median environmental reference in the existing neural residual

## Rationale

The fixed-leaf conditional median improves the environmental reference's central
K0 MAE0.67% on source validation, with an interval crossing zero and worse Q90
error. This exploratory integration asks whether the retained tail-aware native
DOC residual can use that central signal while recovering high concentrations.
It does not declare the prior reference experiment successful or adopt its bare
tree as the complete model. Source partitions142/143/144 and seeds42/43/44
remain the development population; geography/external queries stay unused.

## One change to the retained full procedure

Replace its log-mean environmental reference with the fixed conditional median
of source concentrations on the same log-fitted forest partitions. Refit the
existing ecology self encoder, observation GRU, native scalar residual head and
ecological-memory fusion. Keep the current30-epoch/patience5 budget, tail weight2,
learning rates, initialization, readout features and fusion selection method.
Use source train labels and source validation for the existing epoch/scale/memory
selection. No reference-history projection, chemical pretraining, new attention
operator, extra graph layer or quantile selection is introduced.

For source training residuals, use the five nested station-held-out forests
already saved by `doc_reference_trajectory_v1`. Verify dataset/mask/parent identity,
tree settings and each complement of held stations before reuse. Rebuild the
fold-hidden47-column features, populate distribution leaves only from that
fold's source training labels, and predict the held source DOC cells. Save their
native conditional medians as log1p values in the existing OOF cache interface;
this is storage convention, not a log-space residual loss. No held station can
populate its own source leaf distribution or input history.

Inference uses the full source-trained median reference. Fit all new readout
normalization from source OOF predictions. Save the refitted neural stage before
memory export, with resumable OOF folds, configuration and execution sources.

## Products and interpretation

Keep old complete/residual/tree predictions and add median reference, median
plus neural residual, and median plus neural/memory fusion on the identical K0
source validation cells. Report MAE, Q90, signed bias, station/partition/seed
effects and5,000 paired station draws. Replay saved neural/fusion states and
verify OOF fold exclusion, raw information equivalence and train-only readout
normalization. Run existing tests/lint/audit and inspect figures.

Compare the integrated procedure with the actual retained complete model, not
only its changed environmental reference. A gain over the median tree alone
does not establish an upgrade. The results determine whether this mechanism
should receive further source development; an uncertain small source gain is
not a reason to repeat old geographical/external tests. Keep one complete model
procedure and the unsuccessful readouts as recorded controls.
