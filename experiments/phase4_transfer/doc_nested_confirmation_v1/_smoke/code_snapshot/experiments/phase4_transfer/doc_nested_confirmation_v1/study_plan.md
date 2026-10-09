# Fresh station-role confirmation of separate chemical DOC calibration

Date: 2026-10-04. Model recipe fixed before this study's target evaluation.

## Research question

Does a separate chemical increment preserve useful chemistry information while
improving the stability of sparse DOC station calibration?

The previous 242/243/244 confirmation found K0/K3 reconstruction gains for the
chemistry-aware procedure, but the joint four-coordinate support fit and
ecological selection weakened K5. A subsequent source-validation development
experiment kept the legacy temporal fit and ecological mix fixed and added
only a separately regularized chemical2 increment. Conditional station CV
improved MAE by 0.635% at K3 and 1.234% at K5 versus the legacy chemistry
procedure (K5: 3/3 partitions, 9/9 fitted packages), with tail gains as well.
It did not yet outperform the current joint procedure's validation curve.

This study directly compares the two adaptation methods on new held-station
role assignments after refitting the complete model pipeline.

## Fixed design

- Cohort: the existing ST357 Mississippi dataset, 357 stations and 654 months.
- New station-partition seeds: **342/343/344**, generated from observation
  availability and the existing value-blind random-station protocol.
- Training seeds: **42/43/44**. Nine complete fresh fitted packages.
- Target: DOC only; pH and specific conductance remain measured auxiliary
  inputs with the existing monthly alignment and absent-chemistry fallback.
- Source/validation/target roles: 232/54/71 stations per partition.
- K: **0/1/3/5** retrospective support observations per held station. Preserve
  the existing nested five-candidate support schedule. Exclude all five
  reserved candidates from query at every K. Support can postdate a query.
- Refit source forests, station-blocked context OOF, temporal/backbone neural
  stages, ecological profile, legacy temporal basis, chemical decoder, chemical
  PCA and all source-validation choices. No fitted parent from earlier station
  role assignments is reused.
- Preserve the previous retained training budgets and recipes. Source forests
  are OOF; the complete source neural pipeline is not fully OOF.
- The retained spatial path has no river messages; this experiment evaluates
  chemical/history information and station adaptation.

## Separate chemical increment

After the complete legacy chemistry-integrated prediction, fit the remaining
centered log1p support residual on two source-fitted chemical coordinates.
Keep the legacy support alpha/ridge and ecological gamma fixed. Use chemically
active support rows only; add no second intercept.

Coordinate center/scale come from active months at source-training stations.
Source validation chooses one rule across K3/K5, weighting K and station
equally. Candidate ridge: **0.1/1/10/100**; strength: **0/0.25/0.5/1**. Ties prefer
zero correction, smaller strength and stronger ridge. K0/K1, unavailable query
chemistry, fewer than two active support observations, and zero increments
preserve the complete legacy predictor exactly. No hierarchical prior, new
kernel, larger model or new decoder is introduced in this confirmation.

The matched availability-only increment uses the same operator and candidate
grid. Preserve conditional five-fold validation diagnostics (fold seed
`4100 + partition`) separately from the final target result.

## Six fixed comparison curves

1. General model: `point_integrated_legacy`.
2. Current joint chemical procedure: `neural_chemistry_integrated_selected`.
3. Chemical decoder with legacy calibration: `neural_chemistry_integrated_legacy`.
4. Chemistry-aware trees: `tree_chemistry_selected`.
5. Separate chemical increment: `neural_chemistry_integrated_nested`.
6. Separate availability-only increment: `neural_chemistry_integrated_nested_masks`.

All six curves are reported at every K. Representation/checkpoint/calibration
choices use source validation. Only reserved target support DOC enters station
adaptation. Target query DOC is added at the final evaluation export.

## Evaluation

Primary question: paired native MAE change for the separate increment versus
the current joint procedure at **K5**, with **K3** reported alongside. Keep the
same comparisons against legacy calibration, the general model, chemistry
trees and the availability-only increment. K0/K1 are exact behavior controls.

Report MAE, relative reduction, RMSE, R², log-space error, source-derived Q90
tail MAE/bias/recall, correction sizes, chemistry availability and station gain
and harm. Average seeds within partition, then weight partitions equally. Use
5,000 paired whole-station bootstrap draws with shared multiplicity for
stations repeated across partitions. Intervals are descriptive, without
multiple-comparison adjustment. Seeds and role repetitions are not additional
ecological samples.

Report all three partition and nine package directions. Retain unsuccessful
combinations; do not select a best K, station or method using target results.
Chemistry availability at observed query cells and genuinely missing DOC cells
is shown separately. This is another **ST357 station-role replication**, not
external-basin validation or a prospective forecast.

## Completion and next decision

Replay all nine saved fitted pipelines and final curves, run complete analysis
and inspect its figure, then record the scientific result. If the separate
increment improves K5 and retains the useful K3 behavior, it becomes evidence
for the station-calibration upgrade. If it does not, preserve the existing
model and use the outcome to motivate population-level chemical shrinkage or
tail modeling in a separate study. This round does not retune after seeing its
target outcomes.

Old results stay intact. New predictions, component states, settings and
reports live in this directory; large fitted forest/representation caches
remain local. A short smoke run tests execution only and is excluded from the
scientific confirmation.
