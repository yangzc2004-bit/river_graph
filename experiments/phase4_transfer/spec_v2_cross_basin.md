# Phase 4 cross-basin multi-analyte specification (v2)

Status: **versioned protocol freeze, 2026-09-24**. This specification is a
new route after the observed Stage-2A diagnostic and the target-unseen K=0
output-scale blocker. `spec_v1.md` and its addenda remain immutable. This
file does not reclassify prior exploratory results as preregistered evidence.

## Task definition

For each target analyte `a` in `{doc, ph, spec_conductance}` and each target
HUC6 `b` in the five ST357 leave-out tasks:

1. Source HUC6 basins are all HUC6 basins outside `b`. The target analyte's
   labels are visible in source basins for fitting, source validation, output
   scaling, and source-only model selection.
2. All outcome labels in target HUC6 `b` are hidden from fitting, early
   stopping, normalization, threshold selection, and model selection.
3. The frozen largest-component rows define the target task. The query cells
   are fixed before evaluation and shared across K.
4. Nested target-analyte support sets reveal exactly K cells for
   `K in {0,1,3,5}` during adaptation. At K=0 the model uses no target-HUC6
   outcome label; at K>0 only the declared support labels are visible.
5. Query labels are opened once for final scoring and never enter prediction,
   support construction, threshold choice, or model selection.

Target-HUC6 ecological, hydroclimatic, temporal, and graph inputs may be used
as covariates. Target-HUC6 outcome labels for other analytes are also hidden
from the primary task; a protocol that exposes them would be a separately
versioned multi-output task.

The primary task is **same-month spatial HUC6 leave-out**. Existing E1/E2a/E2b/E3
families may be reported only as secondary analyses after their analyte
specific visibility and sample-size rules are frozen; they cannot replace the
primary same-month task after results are seen.

## Recorded run roles

Every task records:

```text
source_analytes, target_analyte, source_basins, target_basin,
missingness_family, K, support_cells, query_cells, visibility_role,
seed, dataset_hash, mask_hash, config_hash, runtime_snapshot_hash
```

The visibility role must state whether a row is `source_fit`, `source_val`,
`target_support`, `target_query`, or `target_covariate_only`. A target query
label must never appear in a source fit or validation artifact.

## Baselines and information decomposition

Before any feasibility model is selected, run the following under identical
tasks and visibility rules:

- source-analyte month-of-year climatology;
- local support mean and mean-bias correction where K>0;
- fixed analytic support blend;
- EcoRandomForest ecological/time baseline;
- single-analyte H2X;
- matched no-graph and no-ecology controls.

The baseline ladder and deterministic tie order are frozen before target query
labels are opened. If a strongest simple baseline is used for a relative
endpoint, it is selected from source-only validation episodes. Target query
error cannot choose a baseline, mask, analyte, basin, or model.

The primary information decomposition reports temporal inputs, ecological
inputs, local support, and graph structure as matched additions. The no-message
control is retained as a topology ablation; a model name does not establish a
topology contribution.

## Model ladder

1. The fixed baselines above.
2. A universal base with a shared ecology--time representation and an
   analyte-specific native-unit output head. It is trained on source HUC6
   labels and produces a legal K=0 prediction for the held-out target HUC6.
3. A linear or analytic residual adapter using only the K target support cells.
4. A task-conditioned GNN only if Stage 2 and the feasibility pilot show a
   stable signal. It may contain a shared graph/ecology encoder, an analyte
   task representation, and a support-set encoder. At K=0 its support module
   must bypass exactly to the base prediction.

The historical `support_encoder_v2` implementation is a failure comparator,
not a default implementation target. New models must test query-only
shortcuts, support-value shuffles, support-site shuffles, target-HUC6 label
leakage, and K-curve degradation.

## Primary estimator and endpoints

For every analyte and target HUC6, calculate native-unit MAE after averaging
query cells within calendar month. Then average months equally within target
HUC6 and target HUC6 tasks equally within analyte. Report pooled analyte
results and every analyte/HUC6 cell separately; do not pool native MAE across
the three units.

Primary endpoints:

- paired Delta-MAE for K=5 versus the source-only-selected strongest simple
  baseline;
- relative MAE reduction for the same comparison;
- K curve for `{0,1,3,5}`;
- support value-shuffle and support site-shuffle gaps;
- cluster-bootstrap 95% intervals using calendar month as the shared cluster,
  with HUC6 as fixed strata.

Secondary endpoints are log-space diagnostic error, source-derived Q90 tail
error, high-value recall, coverage, and interval width. Q90 thresholds are
derived from source labels only and are frozen before query scoring. Fewer
than 20 unique query cells is marked unstable.

## Feasibility and confirmation gates

The feasibility gate requires all of the following:

1. K=5 improves the selected simple baseline by at least 10% for at least two
   of the three analytes.
2. The improvement direction is present for at least three of five target
   HUC6 tasks.
3. K=5 is no worse than K=0 and K=1 for the primary aggregate.
4. True support beats both value-shuffled and site-shuffled support.
5. No primary analyte/HUC6 cell worsens by more than 5% while pooled success
   is claimed.

The confirmation stage uses five seeds for K=0 versus K=5 and three seeds for
the full K curve, with masks and estimators unchanged. No external data may
change the model, endpoint, or K ladder.

## Uncertainty and external case

Only val-only empirical validation calibration is permitted. Coverage and
median interval width must be reported together; width inflation above 2x is
an operational limitation. Positive error enrichment alone does not unlock a
blind-spot or active-sampling claim. A new ranking gate must be passed by at
least two tools or indicators.

The external basin remains **future pending**. It must be selected by the
existing availability and provenance rules before model results are inspected
and evaluated once with the frozen model and scripts. The external case is
not a substitute for the primary ST357 HUC6 task.

## Provenance and artifact boundary

Each product row contains at least:

```text
analyte, basin, month, station, y_pred, support_count,
uncertainty, model_name, visibility_role
```

Parquet outputs and sidecars bind dataset, node, edge, mask, config, spec, and
runtime hashes. New artifacts belong under
`experiments/phase4_transfer/`. Existing Phase 0--3 and transfer v1 files are
not overwritten.

