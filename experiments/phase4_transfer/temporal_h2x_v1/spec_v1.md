# H2X-T temporal extension specification v1

## Scope

This experiment extends the released H2X `TransportGCNImputer` with a causal
12-month GRU over the spatial hidden representation.  DOC, pH, and specific
conductance are fitted as three separate single-analyte models.  The existing
H2X snapshot runner is the matched baseline; this experiment does not modify
the frozen Phase 0--3 artifacts.

## Frozen model and data contract

- Spatial trunk: `transport_enc`, river edges, bidirectional edge direction,
  hidden width 64, two spatial layers, dropout 0.1, ecological encoder on.
- Temporal trunk: one-layer GRU, hidden width 64, lookback 12, causal windows;
  the current month is the final GRU step.
- Inputs: temperature/discharge and masks, month sine/cosine, static
  coordinates, regime ecology, target-value and target-visibility channels,
  and the causal history-valid channel.
- Target transforms: DOC and specific conductance use `log1p`; pH uses
  train-cell standardization on the raw scale.
- Targets: `mississippi_graph_graphfix_st357.pt`,
  `mississippi_graph_ph_st357.pt`, and
  `mississippi_graph_spec_conductance_st357.pt`.
- Masks: `e1_r20_seed42`, `e2a_strict`, `e2b_partial`, and
  `e3_spatial_seed42`.
- The parent masks were frozen on DOC.  For pH and specific conductance, each
  role is intersected with that analyte's own `y_mask`; the target-specific
  mask preserves the parent missingness family while removing cells with no
  target label.  The generated mask file is bound in provenance and is shared
  by H2X and H2X-T for that analyte.
- A window beginning before the first month repeats the first input and sets
  `history_valid=0` for the padded positions.

## Visibility and training contract

Training loss uses a fresh random half split of train cells into context and
loss targets.  Validation labels are used only for early stopping.  Test
labels, future-month labels, and hidden query labels are never placed in an
input channel, standardization statistic, or model-selection calculation.
Full-grid products retain split roles and observed labels only for final
evaluation.

## Staged execution

### T0: interface contracts

The spatial encoder/head split, temporal window builder, analyte loader, and
V3 provenance schema must pass the temporal contract tests.  A lookback-one
spatial forward must match the released H2X forward exactly.

### T1: smoke

DOC, `e2a_strict`, seed 42, three epochs.  The run must emit a finite,
nonconstant full-grid prediction and a V3 sidecar containing the temporal
configuration, target transform, input hashes, runtime snapshot hash, and
single-run identity.

### T2: pilot

The pilot is 3 analytes × 4 masks × 3 seeds × {H2X, H2X-T} = 72 runs.  The
same masks, query cells, metric script, and target units are used for both
models.  The pilot is a feasibility decision: proceed to T3 only if at least
two analytes improve on E2a/E2b on average and no systematic E1 degradation is
present.

### T3: formal H2X-T products

After T2 passes, run H2X-T for all three analytes, four masks, and five seeds
(60 runs).  Report per-analyte and pooled MAE/RMSE/R², Q90 tail error, basin
strata, K-shot outputs where applicable, model disagreement, and a temporal
ablation summary.  No new uncertainty method or architecture is introduced in
this version.

## Provenance requirements

Every run sidecar is version 3 and binds `temporal`, `lookback`, `causal`,
`temporal_hidden`, `target_analyte`, `target_transform`, dataset and mask
content hashes, `runtime_snapshot_hash`, `runtime_code_snapshot_sha256`, and
`run_identity_sha256`.  New products live below
`experiments/phase4_transfer/temporal_h2x_v1/`.
