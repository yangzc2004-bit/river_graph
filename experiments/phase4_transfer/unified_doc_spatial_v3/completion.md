# Supervised few-shot DOC adaptation: completed experiment

## Result

All nine station-partition/seed packages completed. Training the adaptation
representation changed its features and improved source-validation error, but
did not materially improve prediction at held-out stations. The supervised GRU
projection remains close to its PCA control and the matched tree projection.
This iteration does not establish a new spatial-transfer performance gain.

The experiment trained 18 small projections and 45 station-blocked environmental
forests in 331.8 seconds of recorded run time. The environmental and recurrent
experts used for final prediction, and the version-2 fusion, stayed fixed.

## Main comparison

MAE in mg/L, averaged over three training seeds within each of three equally
weighted station partitions. K is the number of available observations at each
target station. Every row uses the same query cells.

| Fixed-fusion adaptation | K = 0 | K = 1 | K = 3 | K = 5 | Q90 MAE, K = 5 |
|---|---:|---:|---:|---:|---:|
| Constant correction | 1.9136 | 1.8705 | 1.6392 | 1.6085 | 6.7368 |
| GRU PCA | 1.9136 | 1.8705 | 1.6345 | 1.5990 | 6.7060 |
| Supervised GRU projection | 1.9136 | 1.8705 | 1.6358 | 1.5987 | 6.6904 |
| Tree PCA | 1.9136 | 1.8705 | 1.6291 | 1.5961 | 6.7122 |
| Supervised tree projection | 1.9136 | 1.8705 | 1.6307 | 1.5952 | 6.7115 |

At K = 5, supervised GRU adaptation improves on its PCA control by **0.0208%**
(joint station-bootstrap 95% interval: **-0.4062% to 0.4428%**). Two of three
partitions and four of nine fitted packages improve. At K = 3 its MAE increases
by 0.0802%. The tree projection changes little too: K = 5 improves by 0.0585%
over tree PCA, with an interval of -0.0726% to 0.1862%.

The supervised GRU adapter has 0.2184% higher K = 5 MAE than its equally trained
tree counterpart; the paired interval also includes zero. The environmental-only
base gives the same substantive result: K = 5 GRU MAE changes from 1.5999 to
1.5995, and the supervised tree adapter reaches 1.5962.

The complete GRU method reduces MAE by 16.46% from K = 0 to K = 5. Most of that
benefit already comes from local level correction, which achieves 15.95% without
a temporal shape term. The new supervised projection must not receive credit for
that existing support-information benefit.

## What the training teaches us

- All nine GRU projections selected a nonzero training epoch. Mean validation
  MAE decreased from 1.7855 to 1.7506 under the training-time evaluation criterion.
  Mean rotation-invariant subspace displacement was 1.1391. The representation
  changed substantially, so the weak test gain is not explained by a frozen or
  inactive optimization path.
- All 18 representations retained effective PCA rank 16. Both branches store
  32 projection coefficients. The orthonormal parameter space has dimension 29;
  because ridge predictions are invariant to within-plane rotations, the
  predictive subspace has 28 degrees of freedom.
- K = 5 source validation enabled a finite shape correction in all nine learned
  GRU packages, compared with seven of nine PCA packages. Selecting a more active
  shape correction did not produce a corresponding material test improvement.
- GRU Q90 MAE decreased from 6.7060 to 6.6904, a 0.23% reduction, with the same
  direction in all three partitions. Its Q90 MAE is also slightly lower than
  that of the supervised tree representation. This is a descriptive tail signal;
  this analysis does not estimate a paired tail interval.

The immediate scientific conclusion is that selecting better directions within
the frozen temporal state is insufficient here. It does not show that temporal
modeling has reached its performance ceiling or identify a unique cause of the
source-to-target gap.

## Next model experiment

Train the existing GRU temporal state itself for support-to-query reconstruction.
Keep the environmental prediction fixed and initially unfreeze only the recurrent
temporal module and its output representation. Use held-station source episodes
so that the state learns variations that a small number of DOC observations can
identify at a new station. Retain the existing constant, PCA and supervised-tree
adapters as controls, and choose checkpoints using source validation.

This changes the representation that is available for adaptation, rather than
continuing to search projection dimensions or regularization values on the same
frozen features. It is the next proposed experiment; no additional backbone
training was started in this version.

## Evaluation scope and reproducibility

These are development results on previously examined station partitions within
ST357. Support dates can follow query dates because the task is retrospective
reconstruction. All five reserved support dates are excluded from query at every
K. There are 10,520 distinct query station-months at 172 distinct target stations;
repeated seeds and partitions do not add independent observations.

The new OOF environmental predictions exclude held-station labels from both
fitting and feature visibility. GRU/tree feature extractors remain frozen from
source training; their weights are not independently refitted for each source
fold. The source-view label hiding and OOF residual construction are recorded
separately. This experiment does not isolate river-message contributions.

Verification completed:

- 546 tests passed, 2 skipped; ruff passed.
- All nine saved query products replayed bitwise from saved components.
- Version-2 constant-control predictions are bitwise unchanged for every K.
- Historical artifact audit exited successfully with its existing documented
  exclusions and partial-coverage record unchanged.
- Earlier experiment directories and manuscript claims were not modified.

Reproduce using the project's uv environment:

```bash
uv run python scripts/run_ladder.py --experiment unified-doc-spatial-v3
uv run python scripts/analyze_unified_doc_spatial_v3.py
uv run python scripts/verify_unified_doc_spatial_v3.py
```

The verifier needs the local frozen expert packages and representation caches.
Large `representations.npz` caches are excluded from Git. Saved projection
parameters, OOF predictions, training traces, query products, and analysis tables
are retained. Full numerical results and paired intervals are in
[`analysis/findings.md`](analysis/findings.md).
