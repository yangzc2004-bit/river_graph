# Validation record

## Reproduction

Run in the repository uv environment:

```bash
uv run python scripts/analyze_doc_river_hourly_response_v1.py
uv run python scripts/plot_doc_river_hourly_response_v1.py
uv run python scripts/plot_doc_river_hourly_response_v1.py --chinese
uv run python scripts/verify_doc_river_hourly_response_v1.py
```

The verifier regenerates 422 paired hours from the preserved XLSX through the
original duplicate policy (two conflicting site-times excluded), checks the
parent source/product receipt, replays three CSV tables, one parquet and the
summary, and checks both language figure receipts. All 15 segments are retained.

## Numerical and methodological checks

- Synthetic one-hour translated pulses retain their widths and recover the
  supplied scale without searching for a lag.
- Plateau peaks produce clock intervals rather than a selected convenient
  timestamp; separated equal maxima are distinguished from one plateau.
- Censored limbs produce missing widths. Missing hours are never bridged.
- High-turbidity maxima are not replaced by low-turbidity points. The samples
  bracketing the crossings are included in the width-quality check.
- Unknown turbidity and constant series remain explicit.
- The real paired source has unique timestamps and no excluded conflict times.
- Peak-amplitude extrapolation beyond the source laboratory calibration range
  is recorded separately from the >600-FNU retrieval regime. It is not used to
  select a new event set or create replacement peaks.

## Visual review

The English and Chinese result figures were opened and inspected. The Chinese
chronological examples and all three complete atlas pages were inspected.
Annotations that overlapped in the width plot were separated. The DOC bars
refer to turbidity of the *width-support samples*, not a width measured in FNU.
Figures preserve all values, mark >600-FNU samples with crosses, and disclose
that threshold screening does not validate optical peak amplitudes. Site colors
and scales are consistent across both languages.

## Repository checks

- Full pytest: **1,222 passed, 2 skipped**, eight pre-existing warnings.
- Ruff: **All checks passed**.
- Historical `audit_artifacts.py --verify`: exit 0; 83 parquet-only verifications,
  one zero-coverage result, and the previously excluded known G0 conflict.
  Historical missing-sidecar and absent-dataset limitations remain unchanged.
- `git diff --check`: pass.
- No training, raw-data overwrite, old experiment overwrite, or changes to the
  original morphology classes or paper endpoints.

The unrelated DOC geographical/model-development worktree changes remain
outside this commit. Newly generated historical audit inventory files are not
included in this research version.
