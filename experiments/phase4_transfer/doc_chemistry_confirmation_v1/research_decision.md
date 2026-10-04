# DOC chemistry confirmation: research decision

Date: 2026-10-04. Status: all nine production packages completed and replayed.

## Decision

The chemistry-aware DOC reconstruction procedure reproduces a modest gain on
fresh station-role assignments: **2.35% lower MAE without target-station DOC
support and 1.70% lower MAE with three support observations**, relative to the
retained general procedure. Both comparisons improve in all three partitions
and have paired station-bootstrap intervals below zero. With five support
observations, the overall advantage does not persist.

The next performance iteration should strengthen **station calibration**, by
separating the chemical increment from the established temporal support fit
and ecological mixing. The evidence favors retaining measured chemistry in
the predictor; it does not favor adding more layers or repeating this fixed
training budget. This confirmation is closed with its complete curves intact.

## Study and inference population

All predictors were refitted on the same ST357 Mississippi cohort using
station-partition seeds 242/243/244 and training seeds 42/43/44. Each partition
has 232 source, 54 validation and 71 target stations. This is a replication
under fresh station-role assignments, **not external-basin validation** or new
field measurements. The three role assignments overlap in station identity.

DOC is the output. Measured pH and specific conductance are auxiliary inputs.
Their calendar-month means need not come from the same sampling instant as
DOC. K counts retrospective DOC support observations per target station;
support may postdate a query. All five reserved support candidates are excluded
from the query population at every K, keeping comparisons aligned.

The retained predictor combines the general environmental/history model,
temporal residual, ecological transfer, nonlinear chemistry decoding and
station calibration. Its spatial encoder uses the self path with **no river
messages**. These results establish chemistry/history/station-adaptation
performance, rather than an additional river-transport contribution.

Metrics first average the three training seeds within each partition and then
weight partitions equally. RMSE and R² are averages of run metrics. The 30
fixed contrasts use 5,000 paired whole-station bootstrap draws, with each
station's sampled multiplicity shared across partitions. Intervals are
descriptive and have no multiple-comparison adjustment; repeated seeds and
role assignments do not increase the ecological sample size.

## Main reconstruction results

MAE in mg/L. “Chemistry-aware neural” means the predefined
`neural_chemistry_integrated_selected` procedure; its checkpoint, support
representation and ecological mixing choices use source validation.

| Support K | General model | Chemistry-aware neural | Chemistry-aware trees |
|---|---:|---:|---:|
| 0 | 1.913371 | 1.868348 | 1.904915 |
| 1 | 1.867621 | 1.810420 | 1.880329 |
| 3 | 1.730884 | 1.701410 | 1.712390 |
| 5 | 1.666368 | 1.670120 | 1.658794 |

| Neural versus general | ΔMAE [95% interval], mg/L | Relative MAE reduction [95% interval] | Improved partitions / packages |
|---|---:|---:|---:|
| K0 | −0.045023 [−0.089347, −0.001410] | 2.353% [0.077%, 4.575%] | 3/3; 8/9 |
| K3 | −0.029474 [−0.047099, −0.012511] | 1.703% [0.741%, 2.655%] | 3/3; 9/9 |
| K5 | +0.003751 [−0.018596, +0.028861] | −0.225% [−1.719%, 1.090%] | 1/3; 5/9 |

The K1 curve also improves descriptively; the fixed main paired contrasts were
K0/K3/K5. At K5, an interval spanning zero leaves the overall difference
unresolved. The neural procedure's own MAE still falls from K3 to K5, so the
finding is loss of its relative advantage, not worsening with every added
observation. Neural-versus-chemical-tree overall MAE intervals span zero at
all three fixed work points; neither has established overall superiority.

## What the chemical information contributes

Matched decoder comparisons retain the same legacy support coordinates and
use equally sized nonlinear heads: no auxiliary inputs, availability masks,
or measured chemistry. At integrated K5, measured chemistry lowers MAE by
0.009720 mg/L versus the no-auxiliary head (0.586%; interval
[−0.018971, −0.000471]) and by 0.009554 mg/L versus masks alone (0.576%;
[−0.018561, −0.000977]). Both improve in all three partitions and all nine
packages. These contrasts support an incremental predictive contribution
from chemical values beyond their availability pattern.

The full K0/K3 gains also include the new decoder and calibration procedure;
they cannot all be assigned to chemical values alone. Most matched overall
decoder contrasts at K0/K3 remain unresolved, although integrated K0 tail
error improves against both matched controls.

Adding chemical coordinates to support calibration is a separate result.
The integrated selected representation improves descriptively at K3 relative
to legacy coordinates, but its ΔMAE interval spans zero. At K5 it increases
MAE by **0.021328 mg/L** ([+0.001566, +0.044964]), or 1.294%, and is worse in
all three partitions. The direct decoder without ecological integration has
an unresolved K5 difference of −0.001213 mg/L. Thus the weak point is the
interaction between expanded support coordinates, calibration and ecological
mixing, rather than a demonstrated failure of the chemical decoder itself.
Mixing choices change in three of nine K5 packages, an association requiring
a targeted component experiment to explain.

The legacy chemistry-integrated curve has K5 MAE 1.648792 mg/L. It remains a
predefined diagnostic comparison. This round does not replace the accepted
procedure with a target-selected K3/K5 splice.

## High DOC and station stability

Q90 is the source-training 90th percentile within each partition.

| K | General Q90 MAE | Neural Q90 MAE | Chemical-tree Q90 MAE |
|---|---:|---:|---:|
| 0 | 7.996624 | 7.817246 | 7.996020 |
| 1 | 7.805459 | 7.670257 | 7.777970 |
| 3 | 7.804913 | 7.746882 | 7.766868 |
| 5 | 7.620865 | 7.601893 | 7.591156 |

At K0 the neural tail gain is established against the general procedure
(ΔMAE −0.179378; [−0.265537, −0.098747]) and chemical trees (−0.178774;
[−0.288462, −0.072935]). K3/K5 tail differences remain unresolved. Tail bias
is still substantial: −6.447541 mg/L at K0 and −6.228826 at K5. Q90 recall
is 0.572 and 0.588 respectively, leaving extreme-value underprediction as a
subsequent research priority.

K3 improves 110 of 175 unique query stations, with 64 worse and one unchanged.
At K5, 101 improve and 73 worsen, but a few large adverse responses outweigh
the smaller gains: five stations account for 49.1% of positive harm. This
motivates stronger calibration shrinkage, rather than a station-specific rule
chosen from these target outcomes.

## Where the chemistry inputs are available

The fixed evaluation contains 9,705 unique query station-months and 11,551
partition-cell occurrences. At least one chemistry input is available for
9,547 unique cells (98.4%). Partition query counts are 4,163/3,064/4,324;
their Q90 counts are 288/441/415, all above the small-sample flag of 20.

| Grid population | Cells with auxiliary chemistry | Total cells | Availability |
|---|---:|---:|---:|
| All station-months | 63,824 | 233,478 | 27.3% |
| Observed DOC | 22,201 | 22,571 | 98.4% |
| Genuinely missing DOC | 41,623 | 210,907 | 19.7% |

The observed-query comparison largely evaluates stations with known auxiliary
chemistry. Most genuinely missing DOC cells have neither auxiliary input.
Their exact general-model fallback is preserved, but accuracy on those
unobserved DOC cells cannot be inferred from the chemistry-rich query subset.
This distinction defines the deployment question for future work.

## Next performance experiment

Keep the backbone, chemical decoder and legacy ecological mixing fixed.
First fit the established two-coordinate temporal support calibration; then
fit a **separately regularized two-coordinate chemical increment** to the
remaining support residual. A zero chemical increment recovers the complete
legacy calibrated predictor. This avoids asking three or five observations
to refit the full expanded representation and mixing simultaneously.

Train/select one shrinkage rule across K3/K5 using source-station support/query
episodes and source-validation station folds. Compare it with the existing
joint-coordinate fit and legacy calibration, including tail error and station
gain/loss concentration. Use fresh target-role assignments for the next
confirmatory evaluation. The present target results motivate this experiment
and remain the completed assessment of the current recipe.

After calibration is improved, prioritize high-DOC underprediction and the
chemistry-absent deployment population. A larger epoch budget is not the
leading intervention: all nine chemical heads learn nonzero corrections, and
source validation favors chemistry in seven of nine packages against each
matched control. The weak performance link is how the available information
is adapted and combined at a new station.

## Execution and verification

- Nine packages completed; summed package training time: about 66.3 minutes.
- Independent saved-state replay passed for all nine, including all 80 query
  panels per package and 233,478 full-grid rows per package. Neural/query
  products replay exactly; forest arithmetic uses a 1e−12 tolerance.
- Source forests use station-blocked OOF predictions; the complete source
  neural pipeline is source-trained, not fully OOF.
- Full suite: 847 passed, 2 skipped, 3 warnings. Ruff passed. Historical
  `audit_artifacts.py --verify` exited 0 with its existing historical exceptions.
- The final vector/PNG figure was inspected after resolving layout overlap.
- Training sources and frozen recipe were retained. Analysis/plotting repairs
  did not retrain a model or select a target-based replacement.

Reproduce from the repository root with the uv-managed environment:

```bash
uv run python scripts/verify_doc_chemistry_confirmation_v1.py
uv run python scripts/analyze_doc_chemistry_confirmation_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_chemistry_confirmation_v1.py
```

See `analysis/findings.md`, its adjacent CSV files and `figures/README.md` for
complete comparisons, denominators and figure sources. Canonical predictions,
small fitted states and metadata are retained in Git; large local forest and
representation caches remain on disk, as detailed in `artifact_storage.md`.
Full saved-state replay requires that local fitted bundle.
