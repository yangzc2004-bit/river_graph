# Unified DOC spatial adaptation, version 2

## Objective

Improve reconstruction at withheld stations by strengthening two parts of the
existing model: how its environmental and temporal experts are combined, and
how a small local support set adjusts the predicted temporal pattern.

The work has two stages. First, fit robust fusion on the nine saved expert
packages. Then develop a support-conditioned temporal-shape adapter, with an
equally adapted ExtraTrees comparison. The forests and recurrent experts remain
the starting point; this is an extension of the current integrated predictor.

## Fixed experiment and references

Use the existing DOC cohort and the nine version-1 packages: station partitions
142–144, each with training seeds 42–44. Each partition contains 232 training,
54 validation and 71 test stations. These partition outcomes have already been
inspected during development; version 2 is a documented model-development
iteration on the same cohort. Outer-test errors do not select methods or
hyperparameters in either stage.

Keep the same support budgets, K = 0, 1, 3 and 5; the same nested support dates;
and the same fixed query cells, excluding all five reserved support candidates
at every K. Support may postdate a query, so the task remains retrospective
record reconstruction. The experts continue to use training-station DOC only.

Preserve the original ExtraTrees and version-1 hybrid predictions as fixed
references, including their separately selected station-level calibrations.
Their aggregate MAEs are 1.9028 and 2.0015 mg/L at K = 0, and 1.6094 and
1.6261 mg/L at K = 5. Version 2 adds new results without replacing these
references. Existing context-only affine and context-plus-local-forest
controls remain available for interpreting the contribution of the temporal
residual.

## Stage 1: robust combination of the saved experts

### Question

Can a fusion fitted to generalize across validation stations retain the useful
temporal correction while avoiding unstable affine adjustments?

### Method

Reuse the saved full-grid environmental context prediction C, local forest
prediction B, and complete local-plus-recurrent prediction R from every
version-1 package. This stage does not refit the forests or recurrent network.

Compare the following fusion families:

- **Environmental identity:** return C exactly.
- **Convex log-space fusion:** combine log(1 + C) and log(1 + R) with a bounded
  weight, including the environmental identity case.
- **Identity-anchored ridge fusion:** fit an affine combination in log space
  while shrinking its coefficients toward the environmental identity
  (intercept 0, context coefficient 1, temporal coefficient 0).

Divide the source-validation stations into five fixed station folds. Fit each
candidate on the fitting folds and score its predictions on the held-out
validation stations. Use these held-station results to select the fusion
family and its settings, then refit the selected combination on all source
validation stations. Use the chosen fusion's held-station coefficients to
construct validation predictions for adapter selection. The designated
support labels adapt each validation station; all source-validation query
episodes jointly select the global adapter settings. The chosen fusion family
also uses all station-fold scores. This is conditional meta-CV with cross-fitted
fusion coefficients, not a fully nested selection procedure.

The fusion implementation uses convex weights from 0 to 1 in steps of 0.05
and ridge strengths {0.01, 0.1, 1, 10, 100, 1000}. Its correction is written as

    z = z_C + b0 + bC * z_C + bR * (z_R - z_C),

with mean squared log-residual loss plus lambda times the squared norm of all
three correction coefficients, including the intercept. This anchors the
ridge solution to C. Pooled held-station raw-scale MAE selects the candidate,
using each validation query cell once. Exact ties favor identity, then convex
candidates in ascending weight, then ridge candidates in descending strength.
The runner records the station-fold assignment and its seed before scoring
outer queries. No architecture is retrained.

This is cross-validation of the combination of already fitted experts. The
saved experts previously used validation labels for checkpoint and model
selection, so these internal scores are development scores conditional on the
saved experts, not an independent estimate of the complete training pipeline.

### Evidence

Report robust fusion against the original environmental predictor and
version-1 hybrid at K = 0 and across the complete K curve. Compare the calibrated
versions on identical support/query cells. Show selected weights, ridge
strengths and station-fold variation so improvements can be connected to
stable expert combination rather than a single validation fit.

The version-1 context-only affine and C+B controls remain historical diagnostic
references. They are not newly matched station-CV ridge controls for version 2.
The planned six-arm study does not fit an additional ridge-C-only candidate;
its same-base shape comparisons directly compare temporal representations at
equal adapter capacity.

## Stage 2: support-conditioned temporal-shape adaptation

### Question

Can the same few local observations identify a useful change in the shape of a
station's predicted DOC record, beyond the constant log-space offset used in
version 1?

### Method

Extend the selected prediction workflow with a small station adapter. Compare
two temporal representations:

- The frozen observation-aware GRU's hidden-state sequence.
- The environmental ExtraTrees model's per-tree log-prediction vectors.

For each representation, center the feature sequence within station, fit PCA
on source-training stations only, retain two components, and whiten using the
source-trained component variances. Apply these fixed projections to
validation and test stations. Station centering uses the feature record, not
DOC labels. Both representations therefore have the same two-dimensional
adapter capacity. Extract features with the saved experts' training-only DOC
visibility and original source-derived normalization.

Fit support residuals at their own dates using a ridge shape correction and a
station-level correction. Shared support shrinkage candidates are
alpha = {0, 0.25, 0.5, 0.75, 1}; shape ridge strengths are
lambda = {0.1, 1, 10, infinity}. Infinite regularization supplies the
constant-offset case. Query labels are never adapter inputs.

Select shared adapter settings across all source-validation station episodes,
using cross-fitted fusion predictions and the fixed source-trained shape
bases. Adapter hyperparameters are not selected in an additional nested CV.
At a target station, infer the
local adjustment from exactly its K designated support observations. K = 0
returns the selected unadapted predictor exactly. With only one to five labels,
regularization and any shared shape information must be learned from source
episodes rather than estimated freely from the target query.

Apply a matched shape-adapter family to ExtraTrees, with the same support cells,
query cells, candidate capacity and source-selection budget. Give each arm its
own source-selected settings. Preserve the simpler constant-offset correction
as an explicit comparison. Specify any extra representation supplied by the
temporal expert so its contribution can be distinguished from adaptation
flexibility alone.

The six version-2 arms are `context_constant`, `context_gru_shape`,
`context_tree_shape`, `fusion_constant`, `fusion_gru_shape` and
`fusion_tree_shape`, each evaluated at K = 0, 1, 3 and 5. Preserve the frozen
expert packages and version-1 products. Fitting PCA projections, fusion and
support adapters does not retrain a forest or recurrent architecture.

### Evidence

Compare the shape-adapted hybrid with equally shape-adapted ExtraTrees, the
selected robust fusion with a constant-offset adapter, and the fixed
version-1 references. Show how errors and temporal predictions change across
K, partitions and stations. This directly asks whether local observations
improve the temporal reconstruction rather than only shifting its level.

## Common analysis and delivery

Keep the existing MAE, RMSE, R-squared, log-space MAE and training-Q90 tail
metrics. Average training-seed losses within each partition, then weight the
three partitions equally. Use the existing joint station bootstrap for paired
comparisons, preserving repeated-station dependence and fixed-query identity.

Continue the station-level and descriptive ecological-novelty, upstream-support,
discharge-variability and record-availability analyses. These summarize where
adaptation helps; they do not determine which method is selected.

The fixed version-2 comparison set is:

- Robust fusion with constant correction versus matched environmental
  constant correction, at K = 0 and K = 5.
- Robust fusion with constant correction versus the saved version-1 hybrid,
  at K = 0 and K = 5.
- Each GRU/tree shape arm versus the same base's constant correction, at
  K = 3 and K = 5.
- GRU shape versus tree shape within each base, at K = 3 and K = 5.

`scripts/analyze_unified_doc_spatial_v2.py --root <version-2-run-root> --draws 5000`
produces the paired comparison table, individual-seed and partition directions,
complete K curves, log-space and training-Q90 errors, station responses and a
research summary. It reads the original calibrated ExtraTrees and hybrid as
fixed reference curves. It does not select a model from the comparison table.

Save both stages under this version-2 directory with their selected parameters,
predictions, matched comparisons and concise research interpretation. Deliver
an integrated loading/prediction interface for the selected method and update
the manuscript's method, comparisons and figures after completing the planned
analyses. A result is reported even when it does not improve on the fixed
references; outer-test results are not used to choose another candidate grid.

## Existing implementation to reuse

- `unified_spatial_protocol.support_query_cells`: nested support and fixed
  validation/test queries; reuse the existing masks without regenerating them.
- Saved `full_grid.parquet`, `model.json` and `adapter.json`: C, B, R and the
  original choices for all nine packages.
- `UnifiedDOCReconstructor.load` and `predict_components`: restore experts if
  the adapter needs their frozen recurrent computation rather than saved outputs.
- `StationAdaptedHybrid`, `CalibrationEpisode` and
  `station_residual_correction`: existing fusion/adaptation interfaces and the
  constant-offset reference. Add version-2 behavior without changing version 1.
- `analyze_unified_doc_spatial.paired_cells`, `metric_values` and
  `joint_station_bootstrap`: matched errors, metrics and paired intervals.
- `analyze_unified_doc_affine_control`: reference implementation of matched
  context-only recalibration and C+B component controls.
