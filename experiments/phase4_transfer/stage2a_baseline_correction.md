# Stage 2A correction record (2026-09-24)

## Evidence exposure and withdrawal

Exploratory query results were generated and inspected before this correction.
The following revisions are post-result corrections, not preregistration of
an unseen evaluation. None changes the original transfer charter/endpoints
or the frozen Phase 0–3 artifacts.

### First run: `stage2_baselines_v1`

The first run used full HUC6 rows for tasks instead of the frozen largest
components, used target-analyte labels outside the target basin to form the
base without declaring a privileged label regime, treated task-seed repeats
as independent bootstrap clusters, and derived Q90 from query truth. Its
apparent `stage2_gate_pass=true` is withdrawn.

**The original prediction and metric files were accidentally deleted during
the correction. They were not preserved in an archive.**
`stage2_baselines_v1_exploratory_invalidated/` contains only an invalidation
note. The review transcript records the exposed outputs and the withdrawal,
but cannot replace their exact bytes, hashes, or a reproducible historical
result. Do not describe this directory as a retained copy of the first run.
Any future rerun would be a new artifact and could not repair that historical
provenance gap.

### Second run: `stage2a_same_analyte_v1`

This run is retained on disk. It correctly identified the task as a
same-analyte support diagnostic and moved target tasks to the frozen
components, but still used the **complement of the component** as source
rows. That reopened non-component target-HUC6 labels to the source reference.
Its pooled bootstrap also resampled basins rather than treating the five
basins as fixed strata and sharing calendar-month clusters across them.

The v1 predictions, summaries, and verdict are **invalidated pending v2**.
Retaining their bytes is an audit record, not acceptance as evidence. Do not
overwrite them or use their conclusions to unlock Stage 3.

## Corrected v2 scope

`stage2a_same_analyte_v2/` is a new, descriptive reanalysis. Its required
contracts are:

1. Hide labels on every target-HUC6 row; build support/query only on the
   frozen largest-component rows. Source rows are the complement of the whole
   HUC6 hide set.
2. Declare the use of same-analyte labels at source stations as privileged
   reference information. This is not leave-one-analyte transfer.
3. Freeze exact tasks, nested K support, source-derived Q90, runtime identity,
   and prediction/evaluation provenance; verify those bindings before
   evaluation. Query labels cannot affect predictions or source thresholds.
4. Average query-cell errors within month/seed, seed means within month,
   calendar-month means within basin, and basin means equally within analyte.
5. Bootstrap calendar months jointly across basins, keeping all methods, K
   values, and repeated seed realizations together. Do not resample basins.
   Report unique-month sample sizes without multiplying them by seed count.
6. Retain the previous exposure and failed implementations in every scope
   statement. A corrected interval is descriptive, not a new confirmatory
   evaluation on untouched labels.

The detailed estimator and output requirements are in
`stage2_baseline_spec_v1.md` and `spec_v1_1_addendum.md`. The v2 run is now
complete under `stage2a_same_analyte_v2/`: 15,372 query rows, 1,587 task
realizations, whole-HUC6 source exclusion, exact task and visibility hashes,
and shared-month bootstrap output. Its `verdict.json` remains
`stage2_gate_pass=false` and `stage3_unlocked=false`; it certifies only that
the limited diagnostic is reproducible, not that transfer training is ready.

## Stage boundary and unresolved transfer design

The overall Stage 2 gate is **not evaluated**. EcoRF, single-analyte H2X,
matched no-graph/no-ecology controls, the required missingness regimes, and
the information decomposition remain outstanding. Four analytic support
methods cannot stand in for that stage.

The leave-one-analyte source-only K=0 prediction scale is unresolved. The
privileged climatology used here does not solve it. Do not invent target
statistics or target-specific priors, mix raw values across analytes with
different units, or use target query results to choose the output mapping.
Freeze a valid output/transform and source-only selection protocol before
launching transfer training. Stage 3 remains locked; the existing DOC paper
remains the fallback.
