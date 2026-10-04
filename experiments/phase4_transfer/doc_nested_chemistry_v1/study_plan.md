# Separate chemical increments for sparse DOC station calibration

Date: 2026-10-04. Research stage: source-validation development.

## Question and model

Can a separately shrunk chemical correction improve sparse station adaptation
without disturbing the established temporal calibration and ecological mix?

The completed fresh-role study motivates this experiment: its current chemistry
procedure improves K0/K3, but added support coordinates and integrated selection
lose their advantage at K5. The previous chemical RBF interpolation also failed
conditional station cross-validation. This experiment instead uses a linear,
two-coordinate chemical increment after the complete legacy two-coordinate fit.

Keep the fitted backbone, chemical decoder, ecological memory, gamma and legacy
alpha/ridge fixed. At each support budget, compute the legacy prediction at both
support and query. Fit only centered log1p residuals remaining at chemically
active support observations on the source-fitted chemical coordinates. Use a
separate ridge penalty and multiplicative strength. No additional intercept is
fitted. Fewer than two active support observations, K0/K1, strength zero or
absent query chemistry retain the exact legacy prediction.

## Development experiment

- Parents: completed `doc_chemistry_confirmation_v1`, station partitions
  242/243/244 and training seeds 42/43/44. Reuse saved states; no neural or forest
  retraining during this calibration experiment.
- Outcomes: source-validation DOC only. Do not read parent target prediction
  parquet or extract target DOC labels.
- K: 0/1/3/5 with the existing nested support schedule and fixed validation
  query. Selection uses K3/K5 jointly, with equal K and station weights.
- Corrections: chemical coordinates and matched availability-only coordinates.
  Each uses one global ridge/strength rule across K3/K5.
- Candidate ridge: 0.1/1/10/100. Strength: 0/0.25/0.5/1. Ties prefer zero
  correction, then smaller strength and stronger ridge.
- Coordinate mean/scale: chemically active months at source-training stations
  only. No target-station timeline centering.
- Five station folds, seeded by `4100 + partition`, held identically across
  training seeds and coordinate modes. Fit the added operator on four folds
  and assess the fifth. The frozen parent itself used validation for checkpoint
  and calibration selection, so this is **conditional adapter validation**, not
  complete-model OOF confirmation.
- Preserve the full-validation tuning curves separately from the held-fold
  curves. No test-driven K splice or best-station routing.

Compare against the frozen chemistry-integrated legacy procedure, current joint
coordinate procedure, general procedure and chemistry-tree procedure. Report
native/log MAE, RMSE, R², source-derived Q90 MAE and bias, active chemistry
coverage, correction size, station gain/harm and fold/partition consistency.
Training-seed repeats do not increase the number of ecological observations.

## Next research decision

Use coherent held-fold improvement across K3/K5 and partitions to judge the
separate chemical increment. A tuning gain with flat/adverse held-fold behavior
does not warrant full model refitting. If the linear increment shrinks to zero,
the next small candidate is a two-coefficient population chemical slope prior,
learned on validation-training stations and adapted to support; it requires its
own recorded extension before execution.

If the development result supports further testing, freeze the method and
refit all parent stages on fresh role assignments before confirmation. Reusing
old fitted parents under newly assigned target roles would expose their labels.
The existing 242/243/244 outcomes remain the completed evaluation of the old
method. This study remains within ST357 and does not claim external validation.

## Products

Independent run directories contain parent bindings, two fitted correction
states, candidate/fold choices, validation predictions with sidecars, and
completion records. Store the reusable operator and focused leakage/fallback
tests in new files; preserve earlier results and training sources. The old
kernel failure, new linear correction and any subsequent prior experiment
remain separate evidence records.
