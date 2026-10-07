# Validation record

Completed on 2026-10-07 in the repository's uv-managed environment.

## Analysis checks

The dedicated verifier rebuilds inputs, 410 within-station fits across two
populations, all matched-member fits, geographic bootstrap summaries and
structure-association refits from the saved design. It checks 17 tables and all
observed-month products against their saved values.

- DOC perturbation leaves flow-only references and state assignments unchanged.
- Nonpermitted DOC perturbation leaves all observed inputs unchanged.
- References include measured-flow months without DOC; no DOC gap filling.
- Both paired members align to identical calendar months and the original
  29 covariate-selected pairs remain unchanged.
- Single-HUC4 contrasts retain unavailable geographic intervals.
- Six focused tests cover independently defined thresholds, constant flow,
  a hand-calculated seasonal response, identical-calendar pairing, equal-pair
  bootstrap and rank-deficient coefficient handling.

The final verification result is saved in `verification.json`.

## Repository checks

- `uv run pytest`: **1,120 passed, 2 skipped, 8 warnings**, exit 0.
  Warnings come from existing Torch, interval-overflow and geometry tests;
  this study's six tests introduce no warnings.
- `uv run ruff check .`: **All checks passed**, exit 0 after removal of the
  unused plotting import.
- `uv run python scripts/audit_artifacts.py --verify`: exit 0. Known historical
  excluded G0 conflict, kriging zero coverage and absent historical sidecars
  remain recorded by that audit; the new descriptive fits use dedicated replay.

## Figure inspection

All three figures were rendered in English and Chinese as PNG/PDF; every raster
version was actually inspected. The within-river legend was moved above the
panels to avoid covering concentration points. Contrast intervals and all
prespecified response scales/populations remain visible. Association figures
label the separate block fits and per-SD units.

No historical training code, prediction result or morphology class was changed.
