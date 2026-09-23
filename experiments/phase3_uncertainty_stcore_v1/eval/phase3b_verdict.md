# Phase-3B verdict (2026-09-24) — empirical validation calibration, not conformal

Scored against `docs/paper/phase3_uncertainty_spec.md` §5 (frozen gates) on
the completed product: 3 tools × 5 seeds × 8 masks, 24 full-grid products +
24 seed-ensemble files + 8 agreement merges, every row bound to
dataset/mask/config/runtime/spec hashes, label-perturbation contract green.

## Gate 1 — data & identity: **PASS**

All 120 fits complete; full-grid tree separate from observed-only; hashes
bound; hidden-label perturbation changes nothing (contract tests).

## Gate 2 — interval coverage (endpoints v1): **1 of 3 tools PASS**

Test-cell coverage (station-clustered bootstrap CI in eval/coverage.csv):

| tool | overall | E1 | E2a | E2b | E3 | strata (landcover, pooled) |
|---|---|---|---|---|---|---|
| H2X | **0.843** | 0.823 | 0.923 | 0.925 | 0.809 | 0.820–0.910 **PASS** |
| H2X_nomsg | 0.792 | 0.775 | 0.853 | 0.854 | 0.767 | 0.754–0.793 PASS |
| eco_RF | 0.606 | 0.584 | 0.666 | 0.635 | 0.599 | 0.503–0.640 FAIL |

- Overall gate [0.80, 0.95]: **H2X PASS**; H2X_nomsg FAIL (0.792, marginal);
  eco_RF FAIL (0.606).
- Strata gate [0.70, 0.98] (pooled per stratum, n ≥ 50 hidden cells): H2X and
  H2X_nomsg pass; eco_RF fails in every stratum. (The spec's "major
  ecological stratum" was undefined; this evaluation fills it with the
  natural a-priori choice — dominant land-cover class of regime cols 4:8 —
  made after the run and **not** optimized over alternatives; formalize in
  any spec revision. Row-level detail: eval/strata_coverage.csv.)
- Top5/top10 tail coverage and tail MAE are reported separately
  (eval/coverage.csv); E2a/E2b Q95 remain n=2 unstable samples and are not
  ecological evidence (3C rules).

## Gate 3 — monotonicity (pre-frozen rule): **no tool passes cleanly**

Rates over (family × mask) rows (many network_distance rows inconclusive —
empty tertile, distances concentrated):

| tool | support_count | novelty | network_distance | strong reversals |
|---|---|---|---|---|
| H2X | 0.88 correct | 0.50 | 0.50 (75% inconclusive) | novelty 0.13 of rows |
| H2X_nomsg | 1.00 correct | 0.75 | 0.50 (75% inconcl.) | network 0.50 of rows |
| eco_RF | 0.75 | 0.13 | 1.00 (75% inconcl.) | novelty 0.63 of rows |

Only `support_count` behaves as pre-declared across tools. Strict reading of
the frozen rule (≥2/3 correct **and none strongly reversed**) is not met by
any tool. Recorded honestly: the uncertainty–covariate relationships are
**partially** in the declared direction; this is not a pass.

## Gate 4 — three-tool consistency: **FAIL → Phase 4 and Phase 5 PAUSED**

The frozen rule: "if fewer than 2 tools pass calibration, Phases 4 and 5 are
paused; only reconstruction results and descriptive uncertainty remain."
Exactly **one** tool (H2X) passes the calibration gate. Therefore:

**Phase 4 (blind spots) and Phase 5 (budgeted monitoring design) are
PAUSED.** No stable blind-spot claims can be formed (they require ≥2 tools);
no model-sensitive / stable split is authorized. What remains valid and
usable: the full-grid reconstruction product (H2X central predictions),
descriptive uncertainty, and all per-tool diagnostics in `eval/`.

## Structural cause (fact, not an excuse)

Calibration scores use train+val residuals as frozen (spec §3.4). Train
residuals are in-sample and optimistically small, so `q̂` underestimates
test error — visible in the coverage gradient H2X (0.84) > nomsg (0.79) > RF
(0.61), since RF nearly interpolates train rows. This is a property of the
frozen method, discovered after the run.

**Legitimate path to revisit (requires a formal spec v2, not a silent
change):** recalibrate on out-of-sample residuals only (val-only or
cross-fitted), with an explicit "results already seen" acknowledgement and
both v1 and v2 numbers preserved. Calibration is post-processing of the
stored seed ensembles — no retraining needed. This decision is left to the
route owner; until it is taken, the PAUSE stands.
