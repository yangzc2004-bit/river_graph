# Validation — 2026-10-07

## Reproduction and information boundaries

`verify_doc_river_observed_transport_v1.py` passed:

- Nine input/code sources and all saved runtime snapshot files checked.
- 3,026 eligible connection-month inputs and 15,130 operator predictions
  replayed exactly, including all 11 outer monitoring-system folds.
- Current/previous source channels matched their dataset cells and explicit
  source permissions; calendar bins were aligned without gap compression.
- Perturbing all nonpermitted DOC labels did not change the input population,
  source values, availability ledger or geometric normalization.
- 55 fitted operator states ignored altered receiving truth during inference.
  A full nested-fit perturbation of one held system also left its chosen
  parameters, coefficients, tail threshold and predictions exactly unchanged.
  Legitimately observed upstream input channels were held fixed in this test.
- Nine saved tables reproduced, including the 5,000-draw bootstrap summaries,
  receiver counts and omitted-system sensitivity.
- Single-component class intervals stayed unestimable; receiver-month truths
  repeated across connections were counted only once in coverage products.
- Prediction/config/source-mask sidecar and English/Chinese figure manifests
  checked against current files.

## Meaningful unit tests

Eight new tests cover constant concentrations, zero delay, equal paths,
the branch-minus-mean analytic identity, current/previous-only source access,
receiver-equal weights with different numbers of pairs/months, saved source
preprocessing, nested held-system label isolation, and rejected invalid paths,
weights and duplicate calendar rows.

## Repository checks

Commands ran in the existing uv-managed `.venv`, with single-thread BLAS:

```text
uv run pytest -q                 1107 passed, 2 skipped, 8 existing warnings
uv run ruff check .              All checks passed
uv run python scripts/audit_artifacts.py --verify
                                exit 0
```

The historic audit retains its existing 83 parquet-only verifications, one
known G0 conflict and one zero-coverage kriging artifact. No old artifact was
changed to repair those historical statuses.

## Figures

Both English and Chinese PNGs were actually viewed. Titles, axes, legends,
confidence intervals and form counts were checked. A Chinese footer was
corrected from “independent” to “different” receiver-month observations to
avoid implying temporal independence; both language products and their
manifests were regenerated and verification rerun.

## Scope

No existing neural model was trained or tuned. No old frozen endpoint or
experiment directory was overwritten. Unrelated pre-existing tracked changes
and historical untracked files were excluded from this study's commit.
