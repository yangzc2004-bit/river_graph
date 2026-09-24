# Phase 4 transfer specification (v1)

Status: **frozen for Stage 0, 2026-09-24**. This file specifies the first
cross-analyte, cross-basin route. It does not authorize training until the
Stage-1 data gate is complete.

## Task roles

Each run records:

```text
source_analytes, target_analyte, source_basins, target_basin,
missingness_family, K, support_cells, query_cells, visibility_role,
seed, dataset_hash, mask_hash, config_hash, runtime_snapshot_hash
```

`target_analyte` labels outside `support_cells` are unavailable during target
adaptation. Query labels are used only for the final evaluation. Standardisation
statistics come from source training episodes only.

## Data and splits

- ST357 targets are DOC, pH, and specific conductance.
- The five primary HUC6 targets are held out one at a time.
- The external basin is selected by data availability only and then frozen.
- K is `{0, 1, 3, 5}`.
- Existing E1/E2a/E2b/E3 semantics are reused where a target has enough
  observations; masks are versioned per analyte and never silently shared when
  their observed cells differ.
- Same-month K-shot tasks are a distinct task family from E2a/E2b temporal
  forecasting. The current ST357 audit found zero post-2020 same-month tasks;
  a future-month support claim requires a new time-aware protocol version.
- K-shot tasks use fixed support and query cells, nested support sets, and
  task-month cluster bootstrap.

## Model ladder

1. Climatology, local mean, mean bias, analytic blend, EcoRF, single-analyte
   H2X, and no-graph/no-ecology controls.
2. Universal base: shared ecology--time representation with K=0 exact output.
3. Linear or analytic residual adapter using target support values.
4. Task-conditioned GNN only after stage 2 and 3 show stable signal. It uses a
   shared graph/ecology encoder, a support-set encoder, and an analyte task
   representation. K=0 bypasses the support module exactly.

The existing support encoder is a historical comparator. It is not the
implementation target for the new model.

## Primary endpoints

- Relative MAE reduction at K=5 versus the best simple baseline.
- Paired ΔMAE with task-month cluster bootstrap 95% CI.
- K curve `{0,1,3,5}`.
- Support value-shuffle and support site-shuffle gaps.
- Secondary: log-space error, Q90 tail error, coverage, and interval width.

## Confirmatory gates

The Stage-3 pilot passes only if all conditions hold:

1. K=5 reduces MAE by at least 10% versus the best simple baseline for at
   least 2 of 3 analytes.
2. The same direction is present for at least 3 of 5 HUC6 targets.
3. K=5 is no worse than K=0 and K=1 for the primary aggregate.
4. True support beats both support shuffles.
5. No primary analyte/basin cell worsens by more than 5% while claiming a
   pooled success.

The five-seed confirmation uses the same endpoints and frozen masks. External
evaluation uses the frozen model and scripts with no re-tuning.

## Uncertainty

Only val-only empirical calibration is permitted. Report coverage and median
interval width together. Width inflation above 2x is an operational limitation.
Coverage or positive enrichment alone does not unlock blind-spot or sampling
claims.

## Provenance and outputs

Each prediction row contains `analyte`, `basin`, `month`, `station`, `y_pred`,
`support_count`, `uncertainty`, `model_name`, and `visibility_role`. Parquet
outputs and sidecars must bind dataset, mask, config, and runtime hashes.
New artifacts live under `experiments/phase4_transfer/`; old Phase-0--3
artifacts remain immutable.
