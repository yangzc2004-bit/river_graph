# Stage 2 information decomposition specification (v1)

Status: **versioned metrics-only diagnostic; frozen after the Stage-2C
v1.1 control run and before query scoring**.

This artifact compares already scored Stage-2B and Stage-2C task metrics. It
does not train, refit, or reopen hidden query labels. It is an information
comparison under the frozen cross-basin task, not a causal attribution of
model components.

## Scope and inputs

The only primary missingness family currently frozen for this route is
`same_month_spatial_huc6_leaveout`. The task manifest contains no independent
E1/E2/E3 or temporal extrapolation family. This diagnostic therefore reports
the primary family only and records secondary-family availability as
`unavailable`; it cannot establish a general missingness ranking.

Inputs are:

- Stage-2B `task_metrics.csv`, produced by the deterministic baseline
  evaluator;
- Stage-2C v1.1 `task_metrics.csv`, produced by its final evaluator;
- their manifests, verdicts, and the frozen task manifest.

The evaluator consumes metric rows only. It rejects a metrics table containing
query-label columns and checks the shared task key
`(analyte, basin, task_seed, task_index, month, k)` before any comparison.

## Estimands

All deltas use `delta_mae = comparator_mae - reference_mae`; a negative value
means lower error for the comparator. Errors are native-unit MAE and are
averaged within task-month exactly as supplied by the upstream evaluators.
Results are reported per analyte and target HUC6, with an equal-HUC6 pooled
summary within each analyte. Raw MAE is never pooled across analytes.

The following rows are required:

1. **Temporal reference:** absolute MAE of source month-of-year
   `climatology`. No global-climatology arm was frozen, so no temporal
   incremental delta is invented.
2. **Heuristic ecological grouping:**
   `eco_month_climatology - climatology`. The ecological grouping is the
   dominant (argmax) 13-column regime category used by the baseline; it is a
   heuristic covariate grouping, not an ecological type or mechanism claim.
3. **Support addition:** K=5 minus K=0 for each deterministic support method
   (`local_mean`, `mean_bias`, `analytic_blend`) and each Stage-2C learned
   arm. K curves remain descriptive and use K in `{0, 1, 3, 5}`.
4. **Matched graph control:** `h2x_full - h2x_no_graph` at K=0 and K=5.
5. **Ecology-input control:** `h2x_full - h2_no_ecology` at K=0 and K=5.
   This is explicitly a combined ecology-plus-encoder difference because the
   controls do not hold the encoder architecture fixed. It is not an isolated
   ecology effect.
6. **Net learned covariate reference:** `ecorf - climatology` at K=0, labeled
   as a net learned hydro/time/static/regime improvement with architecture
   and feature differences.

EcoRF K=5 minus K=0 is a negative-control row and must be zero under the
frozen arm definition. A nonzero value fails the evaluator rather than being
interpreted as support value.

## Aggregation and uncertainty

For each pair, rows are paired at the task-month key. Within a target HUC6,
task seeds are averaged within calendar month. The confidence interval is a
2000-replicate bootstrap over shared calendar-month clusters. Pooled
analyte summaries give equal weight to target HUC6 basins. Every estimate
records sample size, bootstrap seed, interval, and whether the interval
excludes zero. No result is selected by its test value.

The diagnostic does not unlock Stage 2 or Stage 3 by itself. It cannot support
causal, mechanistic, universal, blind-spot, or active-sampling claims. Full
stage gating still requires the frozen learned controls, support integrity,
and transfer endpoints.

