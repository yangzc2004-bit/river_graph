# Source-validation pilot products

This experiment evaluates an added support-residual interpolator. Its query
products contain validation stations only; no target performance is computed.
The parent neural model, tree model and selected linear calibration stay fixed.

## Per package

- `distances.json`: source-only bandwidth units and sampled source months.
- `linear_remainder_checks.json`: complete all-K linear support correction,
  preclip remainder checks and reproduction of the parent validation scores.
- `all_candidate_scores.csv`: every interpolation-strength/bandwidth candidate.
- `candidate_station_scores.csv`: each candidate's active/all query error sums
  and counts by validation station, enabling incremental station-fold validation.
- `kernel_selection.json`: parameters selected on all active validation queries.
- `source_validation.csv`: full selected validation curves; these tuning scores
  are optimistic relative to independent evaluation.
- `validation_predictions.parquet` and sidecar: parent, availability-kernel and
  chemical-kernel curves, source-validation truth, correction, donor counts,
  parent representation and configuration.
- `station_responses.csv`: station error and correction summaries.
- `config.json`, `timing.json`, `complete.json`: settings, elapsed time and
  bound product identities.

`analysis/` also reports five-fold conditional validation of the new kernel's
parameters. The parent was already selected with validation information, so
this is not whole-model OOF confirmation. `verification/` independently
reconstructs the distance units, preclip remainders, kernel weights, scores
and choices. Runtime archives preserve executed code. `_smoke/` checks the
execution interface and is excluded from analysis and committed products.

## Reproduction

```bash
uv run python scripts/run_ladder.py --experiment doc-chemical-kernel-v1
uv run python scripts/verify_doc_chemical_kernel_v1.py
uv run python scripts/analyze_doc_chemical_kernel_v1.py
uv run python scripts/verify_doc_chemical_kernel_cv_v1.py
```

Use the saved execution source after future model changes. No existing DOC
training result or full-grid product is overwritten by this pilot.
