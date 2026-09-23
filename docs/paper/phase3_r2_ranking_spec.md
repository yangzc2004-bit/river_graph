# Phase-3B-R2 ranking spec (frozen 2026-09-24)

Status: frozen before R2 runs. R2 answers exactly one question: **does the
uncertainty ranking put genuinely wrong station-months first?** It is not
another coverage test and not a redesign of the uncertainty score. Results
from v1/v2 coverage runs have been seen; every choice below is made before
looking at R2 outcomes. Selection rules never read test DOC — test labels
are joined only at final evaluation.

## 1. Selection score

The R2 selection score is the **v2 `uncertainty` column**
(`full_grid_v2_valonly`), and nothing else. No novelty, distance,
model-error or blend weighting terms may enter the score: adding them would
turn R2 into a redesign of the score under evaluation.

## 2. Thresholds, selection layers, selection unit

Two layers, always reported separately:

- **Operational (deployment) threshold.** The flagging threshold is a
  quantile of `uncertainty` computed from **train/validation-era visible
  information only** (cells with `visibility_role ∈ {train, val, context}`).
  Main operating point: the 90th percentile (top 10% by construction of the
  threshold). Applied unchanged to hidden test cells: `flagged = unc ≥ thr`.
  Sensitivity: 80th and 95th percentiles.
- **Ranking diagnostic.** Within hidden test cells, rank by `uncertainty`
  and take the top 5% / 10% / 20% cells. This layer uses **test uncertainty
  only** (never test DOC) and is a diagnostic of ranking quality — it must
  never be reported as a deployment threshold.

**Main ranking operating point: top 10%.** Top 5% and 20% are sensitivity
analyses. The choice of tool or p is never made from hidden-test results.

**Selection unit: station-month.** The random baseline draws the **same
number of station-months**, with `default_rng(42)` and **200 repeats**;
both layers compare against this baseline.

## 3. MAE enrichment (primary metric)

Primary: log1p absolute error `|e_log1p| = |log1p y − log1p median_pred|`.

    enrichment = (mean |e_log1p| of the selected group
                  / mean |e_log1p| of the random baseline) − 1

Also reported: the same enrichment in mg/L MAE. The random baseline uses
equal counts, 200 fixed-seed repeats (report its MC mean and MC std).
Uncertainty on the enrichment: **station-clustered bootstrap 95% CI**
(resample stations with all their rows; 2000 draws).

## 4. High-DOC risk

- **Q90 is the primary tail indicator**; Q95 is always reported but any
  group with n < 20 true-high cells is `unstable`.
- E2a/E2b Q95 can never be deciding evidence for stable blind spots.
- When high-DOC samples are insufficient, the primary evidence switches to
  **continuous tail error / log1p error enrichment** (same formula as §3,
  restricted to truly-high cells or reported as a continuous curve).

Reported per selected group: high-DOC **recall enrichment**
(recall in group / expected recall under the random baseline − 1) and tail
error enrichment.

## 5. Rank association

Per tool × scenario family: Spearman rank correlation between `uncertainty`
and `|e_log1p|` over hidden test cells. Diagnostic only.

## 6. Cross-tool risk direction (criterion input for blind spots)

High-risk = in that tool's top-10% diagnostic set. `joint_high_risk` = cells
flagged high-risk by ≥ 2 of the 3 tools. Report the hidden error of the
joint set vs its complement (per tool's own error). Regions where tools
disagree stay `model-sensitive` (never core evidence).

## 7. R2 acceptance gates (frozen before the run)

Must-satisfy:

1. test DOC is read only in the final evaluation step;
2. uncertainty thresholds and top-p selection rules never depend on test DOC;
3. identical computation rules for every tool, scenario and mask;
4. no tool or p is selected from hidden-test results.

Scientific judgment (recommended rule, frozen here):

A tool has **ranking value** at the main operating point (top 10%, ranking
diagnostic) if:

- the selected group's absolute error exceeds the random baseline
  (enrichment > 0);
- the station-clustered bootstrap CI of the enrichment excludes 0
  (lower bound > 0);
- the direction is consistent in **≥ 2 major scenario families**;
- when high-DOC samples are sufficient (n ≥ 20), high-DOC recall moves in
  the same direction; otherwise continuous tail error is the evidence.

No arbitrary 20% enrichment hard gate: 20% is the Phase-5 sampling-design
gate. R2 judges by direction, CI and cross-scenario stability only.

Stable blind-spot criteria (unchanged): (1) ≥ 2 tools pass R1 coverage;
(2) ≥ 2 tools agree on high-risk direction in R2; (3) the high-risk group
truly shows higher hidden error or higher high-DOC fraction;
(4) model-disagreement regions remain `model-sensitive`.

## 8. Route after R2

- **R2 passes** → Phase 4 may proceed, restricted to: regions jointly
  high-risk under ≥ 2 tools; intersections of support / novelty /
  uncertainty / high-DOC risk; model-sensitive regions listed separately.
  Phase 5 still waits for Phase-4 basin-level stability.
- **R2 fails** → close the stable-blind-spot and active-sampling claims;
  keep DOC reconstruction, empirical uncertainty calibration, the
  coverage–width trade-off, and the negative ranking result.

Evaluation artifacts go to `experiments/phase3_uncertainty_stcore_v1/eval/`
(`r2_ranking.csv`, `r2_ranking_report.md`, `r2_verdict.json`) and are
recomputable by `scripts/run3b_r2_ranking.py`.
