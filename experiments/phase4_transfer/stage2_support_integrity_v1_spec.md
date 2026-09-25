# Stage-2 support-integrity audit (v1)

Status: **protocol freeze before execution, 2026-09-25**.

This is a separately versioned, bounded audit of the already completed H2X
Stage-2C pilot.  It does not change the Stage-2 endpoint, erase the DOC
no-harm failure, unlock Stage 3, select a model, or authorize a new transfer
architecture.  It asks only whether the observed K-shot response is specific
to the declared target support cells.

## Frozen comparison

- Primary model: `h2x_full` (`transport_enc`, river edges, both directions,
  ecology encoder), with the exact Stage-2C v1.1 training budget and seeds
  42, 43, and 44.
- Analytes: DOC, pH, and specific conductance.
- Target HUC6 tasks: the five frozen tasks in
  `cross_basin_tasks_v1/manifest.json`.
- K ladder: `{0, 1, 3, 5}`.  K=5 is the only primary support-integrity
  comparison; lower K values are descriptive.
- Queries, source masks, target rows, and all visibility roles are inherited
  byte-for-byte from the frozen Stage-2C task manifest.
- The model is fitted once per analyte × target basin × seed using source
  train/validation labels only.  Target query labels are never passed to fit,
  early stopping, scaling, or prediction.

## Controls

### True support

The declared support cells and their target-analyte values are exposed at
inference.  The source-only K=0 prediction is retained as a descriptive
reference.

### Value-shuffle

The support locations remain fixed.  The K support values are permuted by a
deterministic seed derived from the unit, task index, and K.  K=0 and K=1 are
marked intrinsically non-identifiable for this control; they are not treated as
evidence.  The shuffle manifest stores only the permutation indices and no
query values.

### Site-shuffle

The support values remain paired in source-support order but are placed on a
deterministically sampled set of alternative, observed target-basin stations
from the same calendar month.  Query cells and true support cells are
excluded.  If fewer than K eligible alternative stations exist, the task is
marked `not_identifiable_by_design`; the runner must not fall back to the true
support locations.  K=0 and K=1 are not identifiable for this control.

The destination manifest is constructed from observation masks only.  It is
frozen before model execution and contains no target values.

## Primary readout

After the label-free product audit succeeds, target query labels are opened
once by the evaluator.  For K=5, report paired query-cell and task-month MAE
for true support versus each identifiable shuffle, with HUC6-stratified,
calendar-month-clustered bootstrap intervals.  Report:

- true-minus-shuffle delta MAE (negative favors true support);
- relative error reduction;
- number and fraction of identifiable task-months;
- every analyte × HUC6 cell and equal-weight pooled analyte summaries.

An apparent support-integrity signal requires a negative pooled delta with a
95% interval excluding zero and at least three of five HUC6 tasks in the same
direction.  This audit cannot override the existing Stage-2 no-harm failure:
even if both shuffle comparisons pass, `stage2_unlocked` and `stage3_unlocked`
remain false until a new, separately frozen decision is made.

## Provenance and failure rules

Every unit binds the task manifest, dataset, node/edge metadata, this spec,
shuffle manifest, model config, runner, and runtime snapshot hashes.  Products
must contain no target labels or derived error columns.  Missing units,
non-finite predictions, duplicate task keys, altered visibility flags, or a
shuffle destination outside the frozen contract fail the audit.  A low
identifiable fraction is reported as a design limitation, not imputed or
silently dropped from the provenance record.

