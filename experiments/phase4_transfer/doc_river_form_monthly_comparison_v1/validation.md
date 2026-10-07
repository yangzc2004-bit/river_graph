# Validation record

Completed on 2026-10-07 in the repository's uv-managed environment.

## Analysis replay

`uv run python scripts/verify_doc_river_form_monthly_comparison_v1.py`
completed with exit 0 and status `passed` after final figure regeneration.

- Reconstructed all 18,688 permitted observed source station-month rows.
- Replayed 40 source-fitted background states and all 112,128 prediction rows.
- Recomputed eight analysis tables with 5,000 HUC4 bootstrap draws.
- Confirmed the original matched pairs and identical calendar months within
  each pair; excluded unmatched observations from paired responses.
- Perturbed held-region DOC and refitted its background: receiving-region
  predictions and fitted source states stayed unchanged.
- Perturbed nonpermitted DOC: analysis inputs stayed unchanged.
- Confirmed that single-region comparisons retain unavailable geographic
  intervals, and checked both language figure manifests.

Seven focused tests cover station weights, hand-calculated responses and
denominators, held-label isolation, constant probability fits, duplicate
observations, shared calendar alignment, and one-region interval handling.

## Repository checks

- `uv run pytest`: **1,114 passed, 2 skipped, 8 warnings**, exit 0.
  The warnings are from existing Torch, interval-overflow and geometry tests;
  the new monthly-form tests introduced no warnings.
- `uv run ruff check .`: **All checks passed**, exit 0, including the final
  figure-label changes.
- `uv run python scripts/audit_artifacts.py --verify`: exit 0. The historical
  inventory retains its known excluded G0 conflict, zero-coverage kriging
  artifact, and historical missing-sidecar/dataset limitations. This audit
  does not replace the new study's dedicated replay above.

## Figure inspection

All three scientific figures were rendered as English/Chinese PNG and PDF.
Both language raster versions were actually inspected. Final revisions retain
zero-frequency points within the scatter axes and explicitly label each
morphology block as a separate addition to the same hydro background.
The contrast figure reports every prespecified population together, rather
than selecting the population with the most favorable effect.

No neural training, existing-model tuning, old result replacement or
DOC-dependent morphology reclassification was performed in this study.
