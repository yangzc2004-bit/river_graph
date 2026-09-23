# Phase-3 uncertainty product spec v2 (frozen 2026-09-24)

Status: **method revision made AFTER v1 results were seen** — stated here
explicitly, per the R0 review. v1
(`docs/paper/phase3_uncertainty_spec.md`) is **not** superseded: v1 and v2
products and numbers stay side by side permanently. This document freezes
the val-only recalibration (3B-R1) and its acceptance gates. Terminology:
this remains **empirical validation calibration** — never "conformal" and
never "coverage guarantee". Validation data participated in early stopping,
so the usual split-conformal requirement that calibration data play no role
in model fitting is not met (cf. Angelopoulos & Bates, arXiv:2107.07511).

## 1. What changes relative to v1 (and what does not)

Change (the only algorithmic change):

- **Calibration scores use `val` residuals ONLY.** Train and context labels
  are excluded from the calibration set. E2b `context` remains legal
  *prediction input* (scenario visibility is unchanged) but its residuals
  are never calibration samples.

Unchanged (must verify bit-identical after the run):

- central prediction (5-seed median), seed spread `s_eff` incl. its floor,
  `support_count`, `network_distance`, `ecological_novelty`,
  `model_agreement`, scenario visibility, log1p calibration space, the
  scaled form `U(x) = q̂ · s_eff(x)`.

Only `q̂`, `pi_lower`, `pi_upper` and `uncertainty` may change.

## 2. Roles recorded per product

Every v2 product records the four roles explicitly:
`train` (fit labels; excluded from v2 calibration),
`context` (E2b inference input only; excluded from v2 calibration),
`val` (early stopping + **the v2 calibration set**),
`test` (hidden; never read by calibration or inference).
`n_cal` = number of `val` cells with observed labels.

## 3. Strata (unchanged definition, corrected wording)

Land-cover-dominant strata (argmax of regime cols 4:8) with n ≥ 50 hidden
cells. These strata are **post-hoc defined, NOT optimized** — the v1 wording
"a priori" is corrected here and nowhere else.

## 4. Network distance grouping (design revision, post-results)

For any future uncertainty-structure evaluation, `network_distance` is
binned as fixed physical groups **{0, 1, ≥2 hops, unreachable}**. Empty
groups or lack of contrast are recorded as **not identifiable** and never
converted into a pass. (The v1 tertile binning is structurally degenerate:
a visible station's distance to the nearest visible station is 0 by
definition.) **R1 does not recompute monotonicity as new supporting
evidence**: val-only changes only the uniform scale q̂ and cannot change
rankings or directions.

## 5. Storage and provenance (v2 products)

Directory: `experiments/phase3_uncertainty_stcore_v1/full_grid_v2_valonly/`
(separate from `full_grid/`; v1 products are never touched). Each v2
product binds, in `provenance_hashes`:

1. the corresponding **v1 product file sha256**;
2. the **seed-ensemble file sha256**;
3. **this spec's sha256** (v2);
4. a **calibration-cell definition hash** (sha256 of the canonical JSON
   `{"rule": "val_only", "keys": ["val"], "context_excluded_from_calibration":
   true, "train_excluded_from_calibration": true}`);
5. the **runtime code snapshot sha256**.

## 6. R1 report requirements (per tool × scenario family)

- coverage with station-clustered bootstrap CI;
- interval width: median / IQR / P90;
- v2/v1 coverage change and v2/v1 width change (both directions reported);
- standardized residual distributions for train / val / test;
- Q90 and Q95 tail coverage and tail width (3C rules: n<20 = unstable);
- `n_cal` (calibration sample count).

Reporting rule: **never report coverage improvement alone.** If a tool's
coverage reaches the gate only by inflating widths past monitoring
usefulness, that trade-off is the result (coverage–width failure mode).

## 7. R1 acceptance gates

Must pass:

1. the 24 v2 products are recomputable by one command;
2. test-label perturbation changes nothing (central prediction, spread,
   covariates, v2 calibration);
3. v1 numbers and files are completely unchanged;
4. v2 calibration reads only `val` residuals;
5. the provenance chain (§5) is complete on every v2 product.

Scientific-result branches (decided by the numbers, not negotiable):

- ≥ 2 tools reach the coverage gate → state **calibration-level improvement
  only**; Phase 4/5 stay locked;
- still < 2 tools → the pause continues;
- significant width inflation → record as a coverage–width failure mode;
  "coverage passed" alone is never a success statement;
- in any branch, monotonicity is not recomputed as new support.

## 8. R2 preview (separate stage; ranking value, not monotonicity)

R2 asks: **can the uncertainty ranking find genuinely wrong and high-DOC
station-months first?** Pre-fixed design: top 5% / 10% / 20% uncertainty
cells; MAE enrichment vs random sampling; high-DOC recall enrichment; rank
association with |log1p error|; cross-tool risk-direction consistency; all
thresholds fixed on validation/training information with `test` used once
for final evaluation only.

Stable blind-spot criteria (frozen): (1) ≥ 2 tools pass R1 calibration;
(2) ≥ 2 tools agree on high-risk direction in R2; (3) the high-risk group
actually shows higher hidden error or higher high-DOC fraction; (4)
model-disagreement regions remain labelled `model-sensitive`.

## 9. Route

1. 3B-R0 — passed (audit closure; contemporaneous identity of the original
   products remains incomplete, 74 gaps recorded);
2. **3B-R1 — this spec: val-only post-processing, no retraining;**
3. 3B-R2 — independent evaluation of ranking value;
4. Phase 4 is discussed again only if R1 and R2 both hold;
5. Phase 5 continues to wait on Phase-4 blind-spot stability.
