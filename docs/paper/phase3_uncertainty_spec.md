# Phase-3 uncertainty product spec (v1, frozen 2026-09-24)

Status: **frozen before any export code runs.** This file decides the
inference-time information sets, the full-grid product format, the
uncertainty/interval algorithm and its calibration space, the three
ecological covariates, and every acceptance gate. Choices marked *frozen
here* cannot be switched after results are seen; changing one requires a new
version with a recorded reason. Endpoints and gates stay in
`docs/paper/primary_endpoints.json` (v1, unchanged). Tools are locked by
`experiments/phase2_ablation_stcore_v1/frozen/phase2_route_decision.md`:
**H2X** (primary GNN treatment), **H2X no-message** (structural sensitivity
control), **EcoRF** (tabular baseline). This phase produces an auditable
uncertainty product — no model performance comparisons.

## 1. Inference-time visibility per scenario (full-grid)

What the model may see in the DOC input channel when predicting the full
grid under each scenario family (labels = observed DOC only):

| scenario | visible DOC labels | notes |
|---|---|---|
| E1 (`e1_r20_*`) | `train` | val/context cells are NOT visible at inference |
| E2a (`e2a_strict`) | `train` | train lies entirely ≤ 2020-12; **no DOC of any test month** is visible |
| E2b (`e2b_partial`) | `train` + `context` | context = the protocol's 20% post-cutoff running-network cells |
| E3 (`e3_spatial_*`) | `train` | every DOC label of held-out stations is hidden |

**Label-perturbation contract (hard).** Perturbing any `test` DOC label must
leave *all* of the following bit-identical: full-grid predictions, the
`uncertainty` column, `support_count`, `network_distance`,
`ecological_novelty`. This is enforced by a regression test
(`tests/test_phase3_uncertainty.py`, step 3 of the build order) before any
product is accepted.

## 2. Full station-month product format

Phase-2 parquets are **observed-cells-only** and must never be reused or
promoted as a full-grid product. The full grid is written to its own tree:

```
experiments/phase3_uncertainty_stcore_v1/full_grid/
    P3_<tool>_s<seedset>__<mask>.parquet      # per tool x mask (8 masks)
    merged/model_agreement_<mask>.parquet     # cross-tool agreement index
```

Grid = 357 stations × 654 months = **233,478 station-months per tool × mask**.
Columns (exact):

| column | meaning |
|---|---|
| `station_id` | station number (string, `site_no`) |
| `month` | month string `YYYY-MM` |
| `prediction_median` | central prediction, **median over the 5 seeds**, mg/L |
| `pi_lower`, `pi_upper` | central 90% interval endpoints, mg/L |
| `uncertainty` | calibrated half-width in **log1p space** (see §3) |
| `support_count` | §4 |
| `network_distance` | §4 |
| `ecological_novelty` | §4 |
| `visibility_role` | mask role of the cell: `train` / `val` / `context` / `test` / `unobserved` |
| `model_agreement` | number of the 3 tools rating the cell high-uncertainty (§6) |
| `provenance_hashes` | json blob: dataset / mask / config / runtime-snapshot / **this spec's** sha256 |

Tools × seeds: H2X, H2X_nomsg, EcoRF × seeds 42–46 (the 3A ensemble). Per
tool × mask the five per-seed predictions are first written to
`experiments/phase3_uncertainty_stcore_v1/seed_preds/` (raw ensemble), then
collapsed into the columns above.

## 3. Uncertainty and interval algorithm (frozen)

Notation: predictions are available in mg/L; convert once to log1p space and
**do everything below in log1p** (frozen: calibration space = log1p; interval
endpoints are transformed back with expm1 and clamped at 0 mg/L — this
choice cannot be switched to mg/L-space calibration after results are seen).
The median is invariant to the monotone transform, so
`prediction_median = expm1(median(log1p preds)) = median(mg/L preds)`.

For each tool × mask, at each grid cell x, with five seeds:

1. **Central prediction**: median of the 5 log1p predictions.
2. **Model uncertainty component (spread)**: `spread(x)` = population std
   (ddof=0) of the 5 log1p predictions; floored at
   `s_floor = quantile_0.10` of spreads over the grid (model-derived, no
   labels involved): `s_eff(x) = max(spread(x), s_floor)`.
3. **Calibration scores** on calibration cells only:
   `q(x) = |log1p y(x) − median(x)| / s_eff(x)`.
4. **Calibration cells** = cells with an observed DOC label in the scenario's
   visible set (§1) — i.e. train (+ context for E2b) **and val**. Validation
   is used both for early stopping and for interval calibration; therefore
   this is **empirical validation calibration**, *not* a strict conformal
   guarantee. The words "conformal" or "coverage guarantee" must not be used
   for this product. (A strict-conformal upgrade — independent calibration
   split or cross-fitting — is out of scope for 3B and would be a separate
   spec version.)
5. **Central 90% interval**: `q̂` = the finite-sample corrected 0.9-quantile
   of the calibration scores, `q̂ = quantile_{ceil((n+1)·0.9)/n}({q(x)})`;
   in log1p: `median(x) ± q̂ · s_eff(x)`; report `pi_lower/pi_upper =
   expm1(·)` clamped to ≥ 0.
6. **`uncertainty` column** := `q̂ · s_eff(x)` (calibrated log1p half-width;
   frozen primary uncertainty scale).

Evaluation cells for coverage = the scenario's hidden **test** cells only.

## 4. The three ecological covariates (frozen definitions)

All three are **scenario-conditional** (computed from the §1 visible label
set) and use **no full-data-fitted** standardizer, PCA, distance basis or
novelty reference:

- **`support_count`**: number of visible DOC label cells (per §1) located at
  stations within ≤ 2 undirected hops of the station in the monitoring graph
  (self included if the station itself has visible labels), counted over all
  months. Local observation support of the neighbourhood.
- **`network_distance`**: undirected hop distance to the nearest station that
  has ≥ 1 visible DOC label (0 if the station itself has one). NaN (with a
  recorded flag) only if no station in the graph has a visible label.
- **`ecological_novelty`**: mean Euclidean distance to the 5 nearest
  "visible" stations (≥ 1 visible DOC label) in the 13-dim standardized
  regime space, where the standardizer (mean/std) is fit **only on visible
  stations' regime rows**. A cell's own station is excluded from its 5
  neighbours.

## 5. Acceptance gates (3B)

**Data & identity**

- 3 tools × 5 seeds × 8 masks all complete; full-grid and observed-only
  trees stay separate;
- every output binds dataset, mask, config, runtime-snapshot and phase3-spec
  hashes (`provenance_hashes`);
- hidden-label perturbation changes nothing (§1 contract).

**Interval coverage** (endpoints v1; each with a **station-clustered
bootstrap 95% CI**, resampling stations with all their rows):

- overall coverage in **80–95%**;
- each major ecological stratum in **70–98%**;
- top5 / top10 high-DOC coverage reported separately (3C rules apply);
- every coverage number ships with its bootstrap CI.

**Monotonicity** (judgment rule frozen here; bins are **train-derived**
tertiles of each covariate computed on the visible set — never re-binned from
test quantiles). Per tool, on hidden test cells pooled per scenario family:

1. mean `uncertainty` non-decreasing across `network_distance` tertiles
   (direction correct if top ≥ bottom);
2. mean `uncertainty` non-decreasing across `ecological_novelty` tertiles;
3. mean `uncertainty` non-increasing across `support_count` tertiles
   (direction correct if top ≤ bottom).

A tool **passes monotonicity** if ≥ 2 of 3 relations are directionally
correct and none is strongly reversed (opposite direction with a top-vs-
bottom gap ≥ 10% of that tool's overall mean `uncertainty`).

**Three-tool consistency**

- each tool reports its own coverage and monotonicity (never select the
  best-looking tool);
- a region is a **stable blind spot** only if ≥ 2 tools pass calibration AND
  agree on risk direction; everything else is `model-sensitive` and excluded
  from core ecological conclusions;
- if fewer than 2 tools pass calibration, **Phases 4 and 5 are paused**;
  only reconstruction results and descriptive uncertainty remain.

## 6. Model agreement

For each cell, each tool classifies **high-uncertainty** = `uncertainty`
above that tool's train-derived median uncertainty (median over cells of
visible stations). `model_agreement` = number of the three tools (0–3)
classifying the cell high-uncertainty. `model_agreement ≥ 2` is the stable
risk-direction signal used by Phase 4's two-layer blind-spot rule.

## 7. 3C high-DOC handling (parallel analysis; never changes 3B metrics)

- **Q90 and continuous tail error are the primary high-DOC analyses**;
- Q95 is still reported, but any cell group with n < 20 true-high cells is
  marked `unstable`;
- seeds never count as new ecological samples;
- tail **coverage** and tail **MAE** are reported separately;
- the current E2a/E2b situation (2 true-high cells, 0 detected) is a
  small-sample fact, **not** an ecological law, and must never be written as
  one.

## 8. Build order (authorization sequence)

1. freeze this spec (**done**, this file);
2. independent full-grid export (own tree, never touching observed-only
   artifacts);
3. label-perturbation + provenance contract tests first
   (`tests/test_phase3_uncertainty.py`);
4. five-seed ensemble generation;
5. interval calibration on train/validation visible labels only;
6. hidden-test coverage evaluation (with bootstrap CIs);
7. Phase 4 blind spots only after 3B passes;
8. if 3B fails: stop active sampling; keep reconstruction + descriptive
   uncertainty only.
