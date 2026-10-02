# Spatial-transfer results for the DOC reconstruction paper

> Evidence update (2026-10-02): this historical summary mixes the 120-tree main product with 300-tree station diagnostics and a station-equal interval. Use [the unified results](spatial_transfer_results_v2.md) and `manuscript_evidence_v2/` for the current manuscript. Its K5 MAE remains 2.023 mg/L; the matching cell-weighted difference is −0.442 (95% CI [−1.026, −0.089]), and 36/43 stations improve. Previously quoted KGML calibration numbers await prediction-level verification.

## Main result

We evaluated a source-regional ExtraTrees expert on the frozen E3 spatial
holdout. The 43 target stations were excluded from model fitting and the same
2,316 query cells were evaluated at every support level. Five target-station
DOC observations were reserved as support candidates before scoring any query.

The regional expert was followed by a station-specific mean residual correction
in log1p space. The correction was shrunk using coefficients selected on the
internal station-heldout split.

| Target-station support K | MAE (mg/L) | Relative reduction |
| ---: | ---: | ---: |
| 0 | 2.466 | 0.0% |
| 1 | 2.320 | 5.90% |
| 3 | 2.160 | 12.41% |
| 5 | **2.023** | **17.93%** |

The K=5 station-clustered bootstrap difference from K=0 is −0.419 mg/L,
with a 95% interval of [−0.820, −0.141]. The same product contains the full
2,531-cell E3 table; the table above uses the fixed paired query so that K is
compared at equal query coverage.

## Mechanism

The support correction improves stations with a large zero-support baseline
bias. Across the 43 target stations, the correlation between absolute K=0 bias
and K=5 MAE reduction is approximately 0.92. This supports a two-part spatial
transfer picture:

1. source-station similarity transfers seasonal and hydroclimatic variation;
2. a small number of target-station observations calibrates the local
   concentration level.

Shuffling support residuals across target stations removes the improvement,
showing that the gain is station-specific rather than a consequence of simply
opening more labels.

## Model comparison

The existing KGML prediction is useful as a structural comparison, but its
current directed message branch does not improve spatial transfer consistently.
The five-seed KGML baseline is about 2.823 MAE on the same type of E3 query;
applying the same support correction reduces it to about 2.544. The regional
expert with support correction remains stronger at 2.023.

Adding a hard support-aware gate, extra geomorphology, alternative residual
forms, auxiliary pH/conductivity profiles, or alternative value-blind support
schedules did not produce a reliable improvement over the regional residual
adapter.

We also compared three regional tree learners under the same nested selection
and support-adaptation protocol. The selected random forest reached 2.048 MAE
on the outer paired query; ExtraTrees reached 2.024 and histogram gradient
boosting 2.086. This reproduces the existing ExtraTrees result rather than
opening a better learner family.

## Recommended figure set

* Main K curve and station bootstrap: `regional_residual_product_v1/k_curve.png`.
* Support-residual mechanism and shuffle comparison:
  `fewshot_residual_paired_v1/analysis/spatial_fewshot_residual.png`.
* Station heterogeneity: `adapter_comparison_v1/station_mechanism.png`.
* River-network spatial map of station-level K=5 gains:
  `station_transfer_map_v1/station_transfer_map.png`.

## Scope of the claim

The result is a historical sparse-reconstruction and spatial-adaptation result
under a fixed E3 holdout. It supports conditional transfer with a small amount
of target-station support. It does not establish that river messages are
universally necessary or that the correction is a prospective forecast before
the support observations are collected.
