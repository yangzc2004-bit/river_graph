# Stage 2B cross-basin deterministic baseline specification (v1)

Status: **frozen 2026-09-24**. This is a no-training diagnostic under
`spec_v2_cross_basin.md`; it does not unlock the feasibility-model stage.
The route and this baseline ladder were written after the observed Stage 2A
diagnostic and the target-unseen K=0 blocker; those prior results remain
recorded evidence, not new baseline claims.

## Visibility

For every frozen `(analyte, target HUC6, seed, month, K)` task, all source
aggregates use the target-analyte `source_fit` labels only. Source validation
labels are used only to select between the two source-only baselines below.
Target support labels are used only by the three support methods and only at
the declared support cells. Target query labels are opened by the evaluator
for final scoring and never enter prediction or selection.

The source split, target component, support cells, query cells, and K ladder
come only from `cross_basin_tasks_v1/manifest.json`. No task is regenerated
from target outcomes.

## Frozen methods

1. **`climatology`**: source-fit mean for the calendar month. If a month has
   no source-fit value, use the source-fit global mean.
2. **`eco_month_climatology`**: source-fit mean for calendar month and the
   station's dominant ecological regime (`argmax(regime)`). Empty groups fall
   back to the source-fit month mean, then the source-fit global mean.
3. **`local_mean`**: mean of the K target support values. At K=0 it falls back
   exactly to `climatology`.
4. **`mean_bias`**: `climatology(q) + mean_s[y(s)-climatology(s)]`. At K=0 it
   falls back exactly to `climatology`.
5. **`analytic_blend`**: a fixed, non-fitted support gate. Let `m` be the
   support mean, `r` its standard deviation plus `1e-6`, and `d` the mean
   undirected graph hop distance from query to support (unreachable support
   uses `d=5`). Define

   ```text
   g = exp(-2.5 * r / (abs(m) + 1) - 0.15 * max(0, d - 1))
   pred(q) = (1 - g) * climatology(q) + g * m
   ```

   At K=0 it falls back exactly to `climatology`. The gate has no fitted
   parameter and reads no query value.

## Source-only selection and reporting

`climatology` and `eco_month_climatology` are scored on source validation
labels for each seed and selected by their mean source-validation MAE within
an analyte/HUC6, with method name as the deterministic tie break. The target
query does not select a baseline. All five methods remain in the product.

The evaluator reports paired K=5 deltas, K curves, log-space diagnostics, and
source-train Q90 tail diagnostics. Tail groups with fewer than 20 query cells
are marked unstable. The artifact is descriptive only: EcoRF, H2X, matched
no-graph/no-ecology controls, and a missingness decomposition remain required
before any Stage 2 feasibility or Stage 3 training decision.
