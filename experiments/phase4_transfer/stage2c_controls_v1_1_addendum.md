# Stage 2C controls v1.1 execution addendum

Status: **retrospective execution record, 2026-09-25**.

This addendum records implementation facts discovered during the post-run
audit. It does not change the task manifest, masks, arms, training budget,
K values, or scientific endpoints, and it does not promote the pilot to a
confirmatory experiment.

## Execution state

- The reviewed plan contains 180 units: 135 GNN fits and 45 EcoRF fits.
- All 180 unit prediction files and selection sidecars were present before
  the scoring evaluator consumed hidden query labels.
- A separate pre-evaluation inventory checked every `(task_index, k, flat)`
  query row against the frozen task manifest, duplicate rows, finite
  predictions, arm/analyte/basin/seed/config identity, and the no-query-label
  flag. It found 61,488 valid prediction rows.
- The runner did load complete label arrays while checking frozen observed,
  support, and query masks. It did not pass target-query values to fitting or
  prediction. The scoring evaluator consumed target-query values only after
  the inventory audit below; this is the relevant no-consumption boundary.

## Visibility wording correction

The source validation labels are used by `GCNDocModel.fit` for its frozen
source-only early-stopping diagnostic. This is allowed by the route protocol:
target-basin outcome labels remain hidden, and source validation labels are not
passed to prediction or target support. The v1.1 specification sentence that
called source validation a diagnostic only was too narrow; this addendum is the
versioned correction. No target query label is used for fitting, early
stopping, normalization, support construction, or prediction.

The product-row value
`cross_basin_target_analyte_source_visible` is retained as a historical
implementation label. It is not one of the route's role enum values and is
therefore reported as a schema limitation. The corrected evaluator explicitly
records this limitation and does not infer a stronger visibility claim from
the string.

## Evaluation boundary

The original v1.1 evaluator remains a descriptive control diagnostic. The
fail-closed `evaluate_phase4_stage2c_controls_r1.py` adds the complete task
inventory, selection-sidecar, finite-value, expected-unit, and finalized-plan
checks before opening query labels. Neither evaluator implements the full
Stage-2 feasibility gate: source-selected-baseline comparisons, support value
and site shuffles, and the information decomposition remain separate required
products.
