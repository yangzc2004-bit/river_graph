# Context-only affine control

Specified on 2026-10-03 (Asia/Shanghai) from an implementation review of the
unified spatial model, before the reviewing agent inspected any formal
confirmation performance. The review identified a specific attribution issue:
the hybrid can fit a validation intercept and slope through log-affine fusion,
whereas the original ExtraTrees arm cannot. A stronger hybrid may therefore
benefit from global recalibration in addition to its second expert.

At the time this note was written (2026-10-02 18:22 UTC), the production process
was live and the first formal run's `complete.json` already existed. This note
does not claim to predate that completion or constitute an original
preregistration. The control was motivated by the model equations, not by an
inspected formal performance result. Reduced-budget smoke results may be used
to check the control implementation.

## Additional ablation

The planned four-arm primary results remain unchanged. This is an additional
single-expert affine ablation, stored separately, with no expert retraining.

For each completed split and training seed:

1. Load the frozen full-grid context predictions and the existing source
   validation station split. Exclude all five reserved validation support
   candidates from the fixed validation query.
2. Fit `a + b * log1p(context_prediction)` by ordinary least squares against
   validation-query `log1p(DOC)`. Select identity or affine by raw-scale
   validation MAE; ties favor identity. Predictions are lower-bounded at zero
   as in the primary hybrid. Reject a nonfinite validation candidate; do not
   use outer-test scores or clip high predictions based on outer-test values.
3. For each K in {1, 3, 5}, independently select shrinkage alpha from
   {0, 0.25, 0.5, 0.75, 1} using the same validation support/query schedule as
   the primary model. Correction is alpha times the station mean of
   `log1p(support DOC) - log1p(affine context prediction at support months)`.
   K = 0 is exact identity. Candidate ties favor smaller alpha.
4. Freeze coefficients and alphas, then predict the outer test support/query
   populations. Support is used only by the retrospective station correction;
   query labels are used only after all selection is complete.
5. Compare the existing hybrid against the selected context-only control at
   K = 0, 1, 3, and 5. For positive K, both models receive their own
   validation-selected station calibration. Additionally report the raw and
   calibrated control against its original ExtraTrees arm, to quantify the
   effect of adding global affine recalibration.

## Estimation and interpretation

Average individual-seed losses within each partition, then weight partitions
equally. Retain cell weighting within a partition. Confidence intervals use
the same joint whole-station bootstrap as the primary analysis, resampling a
station together wherever it occurs across partitions. The planned budgets
remain three station partitions and three training seeds.

The extra control tests whether a second expert adds information beyond
single-expert global recalibration. It does not isolate the GRU architecture
from its ecological features, nor establish river-message gain. These are
random held-out stations in the known cohort and retrospective support
calibrations. They are not independent external basins or prospective forecasts.

## Outputs

`scripts/analyze_unified_doc_affine_control.py` writes only a new
`context_affine_control/` subdirectory in the requested run root, containing
control predictions, all source-validation choices, matched per-run metrics,
joint-bootstrap comparisons, seed/partition consistency, and a report. It
never overwrites primary predictions, fitted expert files, or main comparisons.

Run the smoke check with:

```bash
uv run python scripts/analyze_unified_doc_affine_control.py \
  --root experiments/phase4_transfer/unified_doc_spatial_v1/smoke_v1 \
  --allow-partial --bootstrap-draws 500
```

Run the final additional ablation only after all formal runs complete:

```bash
uv run python scripts/analyze_unified_doc_affine_control.py \
  --root experiments/phase4_transfer/unified_doc_spatial_v1/confirmation
```

## Additional C+B component ablation (2026-10-03)

Added at 2026-10-02 18:28 UTC, after the coordinating agent had inspected the
first formal run's summary. This is an explicitly supplementary component
analysis; it does not change the primary protocol, fitted experts, or primary
four-arm predictions.

The complete temporal expert R consists of local ExtraTrees prediction B plus
a learned recurrent residual. To distinguish the contribution of that residual
from simply combining two differently featured forests, substitute the saved
native-scale `local_pred` B for R and fit a C+B combination.

Use exactly `StationAdaptedHybrid`, the source-validation fixed query cells,
the identity/individual-expert/geometric/log-affine fusion candidate set, and
the same K support schedule and alpha grid as the primary C+R model. The C+B
combination chooses its own fusion and calibration strengths using only that
source validation information. No forest or recurrent model is retrained.

Compare primary C+R against C+B at K = 0 and K = 5, using the same test query
cells, estimand, and joint whole-station bootstrap. At K = 5, each combination
receives its own source-validation-selected station correction. All C+B
candidate scores and chosen coefficients are saved under `local_forest_control`
in `selection.json`; C+B predictions remain in the separate control directory.

This ablation tests the added recurrent residual component within the complete
model. It does not independently attribute effects to the GRU operator,
observation-age features, or river connectivity.
