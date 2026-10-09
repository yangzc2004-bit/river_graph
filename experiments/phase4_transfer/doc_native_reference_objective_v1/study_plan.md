# DOC-aligned environmental reference objectives

## Research question

The retained neural residual minimizes native DOC MAE, whereas its environmental
ExtraTrees reference fits squared error of log1p DOC. Test whether aligning the
reference to native concentration helps unmonitored-station reconstruction,
before rebuilding the existing GNN residual on a changed reference. The source
chemistry warm start and reference-history projection are completed unsuccessful
candidates and are not added to the retained model.

## Fixed source comparison

Use source station partitions142/143/144 and seeds42/43/44, unchanged station-
hidden47-column environmental/hydro/context feature construction and all valid
validation DOC cells as K0 query. Receiving DOC/pH/conductance remain hidden.
Keep the actual retained full model and selected log-space ExtraTrees unchanged.

New reference arms:

1. Native-concentration ExtraTrees: clone each retained forest's configuration
   (300 trees, selected leaf size and feature fraction), changing only the fitted
   target from log1p DOC to DOC. Criterion remains squared error. This is the
   matched target-scale comparison.
2. Log-space L1 gradient boosting: HistGradientBoosting, absolute_error loss,
   log1p DOC target,300 iterations, learning_rate0.05,31 leaves, minimum20
   samples per leaf, l2_regularization1, early_stopping=False.
3. Native-concentration L1 gradient boosting: exactly the same configuration,
   fitting DOC. This is the matched native/log L1 comparison.

No hyperparameter scan or validation-based arm blending. Gradient boosting is
not new to the project: older HGB comparisons used squared-error log targets
and different tasks. This version tests the concentration/loss alignment under
the current whole-station-hidden input regime. Report those differences openly;
it is not an equal-compute comparison with the retained forest/complete model.

Each new arm fits source train labels only. Feature thresholds/bins must be fit
on those rows, and no random held-cell early stopping substitutes for station
validation. Predictions are clipped at zero in native space like current tools.
Record source/validation roles, selected old forest settings, target transform,
objective, fit and product. Save models locally and retain previous products.

## Analysis and model integration

Compare native ExtraTrees with retained log ExtraTrees, and native L1 boosting
with log L1 boosting. Also compare every candidate with the actual retained
complete model, not just the preceding model. Use equal source partitions after
seed averaging,5,000 paired station-bootstrap draws, MAE, Q90, signed bias,
station error and partition/seed direction. Inspect the figure and saved model
replay. Do not infer that a Q90 improvement alone establishes overall superiority.

If a reference improves the source reconstruction, generate its nested station-
blocked OOF targets and refit the SAME retained ecology/GRU/native residual and
memory fusion. Remove every held station's DOC before either fit or prediction
features are made; do not subtract in-sample predictions for the neural training
residual. This is the next integration step, not yet executed or a claimed result.
Geographical/external tests remain outside development. Keep one retained full
procedure; do not pick different winners by region or K.

New precipitation information remains a parallel data-availability task. Its
NASA API access failed in the present environment; this offline objective study
can proceed without weather downloads. No new fitting had started when this
plan was written.
