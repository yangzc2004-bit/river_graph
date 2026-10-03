# Episodic learning for sparse DOC station adaptation

## Research question

Can a representation trained to reconstruct a station from a few local
observations improve on the unsupervised temporal bases used in version 2?
The existing environmental forest and recurrent expert remain fixed. The new
learned component is a small projection of their temporal representations.

Version 2 repaired much of the earlier fusion loss, but its two-component GRU
basis did not outperform a matched tree basis. PCA maximizes feature variance;
this experiment instead trains the basis for support-to-query reconstruction.

## Data and existing models

Use the nine saved DOC packages: station partitions 142, 143, 144, each with
seeds 42, 43, 44. Preserve their source, validation and target stations, the
fixed query cells, and nested support K = 0, 1, 3, 5. All five reserved support
dates remain excluded from queries at every K. The task is retrospective
reconstruction, so support dates may follow query dates. These partitions have
already been examined during model development; the experiment is an explicit
development iteration on the same cohort.

Keep the version-2 regularized fusion unchanged. Apply the new representation
to both its prediction and the environmental forest prediction. Learn one
projection per representation and package, rather than training separate
projections to improve individual outer-test comparisons.

## Source residuals and feature views

Refit five station-blocked copies of the selected environmental ExtraTrees
configuration. Each fold hides all DOC inputs and training labels at its held
source stations. Predict their observed source cells to create OOF residuals.
This uses the actual selected forest settings, including its leaf size.

Extract GRU hidden sequences and per-tree log-prediction vectors from the
existing frozen experts. Represent each source station under its corresponding
fold-hidden input view. Validation and target stations use the original
training-only visibility. Expert weights remain source-trained; these feature
views are not five independently fitted recurrent representations. The
cross-fitted environmental predictions supply the honest source residual
targets for learning the new projection.

## Matched trainable projection

For each representation, center each station's feature record and fit a
source-only PCA whitening transform with 16 dimensions. Use an eigenvalue
floor of max(1e-8, 1e-5 times the leading eigenvalue), and zero unsupported
modes rather than amplify them. Record effective rank and require at least
two supported dimensions.

Learn a 2 by 16 matrix W, initialized to the first two whitened axes. Retract
its rows to an orthonormal basis after every update. Both the GRU and tree
representation therefore have 32 stored projection coefficients and the same
29-dimensional orthonormal parameter space. Orthonormality prevents scaling
the features simply to evade ridge regularization. Retain the fixed first-two
PCA axes as the matched control for each learned representation.

For a station episode, use five support observations spanning its record.
Split chronological observed ranks into five bins, sample one index per bin,
and order them first, middle, last, first quarter, third quarter. K = 3 is a
nested prefix of K = 5; all five are excluded from the episode query set.
Source stations need at least six observed cells. Resample dates each epoch
with the same seeded schedule for the GRU and tree arms.

Fit the existing centered ridge head from support residuals. Train W through
that differentiable solve using native-scale query MAE, averaged over
K = {3, 5} and ridge lambda = {1, 10}, with level shrinkage alpha = 1.
Average source losses equally across station episodes. Use Adam, learning
rate 0.01, batches of 32 stations, at most 100 epochs and patience 15. No
upper DOC clipping is introduced; nonfinite training values are errors.

Select the checkpoint on source-validation queries with the existing fixed
support schedule. Its criterion is pooled query MAE averaged over the same
K/lambda combinations. Include epoch 0 in selection. K = 0 needs no adaptation;
K = 1 has no centered shape information and reduces to level correction.

After fixing the representation, select each final adapter's alpha from
{0, 0.25, 0.5, 0.75, 1} and lambda from {0.1, 1, 10, infinity}, exactly as in
version 2. Fusion validation predictions use its saved held-station
coefficients at both support and query dates. Neither representation training
nor adapter selection receives outer-query labels.

## Comparisons and interpretation

Evaluate two bases, environmental and frozen fusion, each with five heads:
constant correction, GRU PCA, episodically learned GRU projection, tree PCA,
and episodically learned tree projection. Preserve all ten arms at every K.
The budget is 18 small projection fits plus 45 OOF forest fits, using nine
existing expert packages. The recurrent backbone is not retrained.

Primary comparisons are learned versus PCA for each base/representation at
K = 3 and K = 5; learned GRU versus learned tree at K = 3 and K = 5; and each
learned representation versus constant correction at K = 5. Each adapter may
select its own level shrinkage, so these compare complete adaptation methods.

Report native MAE, RMSE, R-squared, log MAE and training-Q90 MAE. Average seed
losses within partition and weight partitions equally. Use 5,000 joint station
bootstrap samples for paired differences. Report training trajectories,
selected epochs, projection changes and support-adapter choices alongside the
errors. The outer test does not choose a projection, epoch, or candidate grid.

A learned-GRU improvement over its PCA control supports task-specific temporal
adaptation; superiority over the equally trained tree representation is a
separate question. Preserve both findings even if they differ. Keep earlier
experiment results unchanged.
