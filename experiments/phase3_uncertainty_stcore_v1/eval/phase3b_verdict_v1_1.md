# Phase-3B verdict v1.1 (auto-generated, 3B-R0)

Original v1 verdict kept unchanged at `eval/phase3b_verdict.md`. Revisions recorded here were made AFTER results were seen and are labelled as acceptance-record corrections, not as fresh gates.

## Calibration gates (coverage)

|           |   overall_equal_mask_mean | overall_gate_080_095   | strata_pooled                                                          | strata_all_in_070_098   | passes_calibration   |
|:----------|--------------------------:|:-----------------------|:-----------------------------------------------------------------------|:------------------------|:---------------------|
| H2X       |                     0.843 | True                   | {'crop_hay': 0.813, 'forest': 0.862, 'urban': 0.868, 'wetland': 0.909} | True                    | True                 |
| H2X_nomsg |                     0.792 | False                  | {'crop_hay': 0.775, 'forest': 0.806, 'urban': 0.778, 'wetland': 0.754} | True                    | False                |
| eco_RF    |                     0.606 | False                  | {'crop_hay': 0.6, 'forest': 0.597, 'urban': 0.626, 'wetland': 0.509}   | False                   | False                |

## Phase 4/5 status: **PAUSED (fewer than 2 tools pass calibration)**

## Calibration diagnostics (standardized scores, in-sample vs out)

| tool      | role   |   score_q50 |   score_q90 |
|:----------|:-------|------------:|------------:|
| H2X       | test   |       2.272 |       6.940 |
| H2X       | train  |       1.996 |       5.549 |
| H2X       | val    |       2.318 |       6.864 |
| H2X_nomsg | test   |       2.679 |       7.875 |
| H2X_nomsg | train  |       1.590 |       5.060 |
| H2X_nomsg | val    |       2.732 |       8.063 |
| eco_RF    | test   |       9.120 |      26.401 |
| eco_RF    | train  |       4.234 |      10.071 |
| eco_RF    | val    |       9.327 |      25.681 |

## Interval widths (mean mg/L, test cells)

| tool      |   mean_width_mg_l |
|:----------|------------------:|
| H2X       |             5.814 |
| H2X_nomsg |             4.605 |
| eco_RF    |             3.287 |

## Corrections to the v1 record

- land-cover strata are **post-hoc defined (not optimized)** — the v1 text calling them a priori is corrected here;
- network_distance tertile binning is **structurally degenerate** (visible stations' distance is 0 by definition): those monotonicity rows are not-identifiable-by-design, never passes;
- provenance: seed files carry retrospective content pins only; see `eval/provenance_audit.json` for the gap list (no post-hoc re-stamping as contemporaneous).

Passing these gates does NOT by itself unlock Phase 4/5; the three-tool rule and the independent ranking-value review (3B-R2) still apply.
