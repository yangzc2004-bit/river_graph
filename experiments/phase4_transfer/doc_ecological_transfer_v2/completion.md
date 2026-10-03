# Support-aware ecological residual transfer: completed DOC iteration

The model now chooses how strongly to use regional ecological residual memory
after target-station observations become available. The four frozen v1 memories
compete with the existing temporal residual; source validation jointly chooses
their mixture and the existing support adapter's correction strength and ridge.
This is an integration update to the existing model. Forests, GRU weights,
ecological profiles and the support basis were retained.

## Spatial reconstruction results

| Model with the existing GRU support basis | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Previous interaction model | 1.835240 | 1.804840 | 1.626925 | 1.581605 |
| Fixed ecological affine memory, v1 | 1.809217 | 1.787986 | 1.621607 | 1.587188 |
| Support-aware ecological affine memory, v2 | **1.809217** | 1.791696 | 1.622073 | **1.579905** |

MAE is in mg/L. K is the number of reserved DOC observations at each target
station. Seeds are averaged within station partition and the three partitions
receive equal weight. Queries are identical for every model and K. Support is
retrospective and may follow a query in calendar time.

The K0 prediction is intentionally identical to v1. Its previously observed
**1.42%** improvement over the interaction model, with a station-bootstrap
interval of **0.52% to 2.45%**, is carried forward rather than counted as a new
replication.

At K1 and K3, v2 improves on the interaction model by **0.73%** and **0.30%**;
their paired intervals are **0.03% to 1.51%** and **0.07% to 0.57%**. Both
comparisons improve in all three partition means. The fixed ecological prior
still has slightly better K1/K3 point estimates than v2, so v2 is not a uniform
improvement over v1.

At K5, v2 removes the fixed prior's mean penalty: MAE changes from 1.587188 to
1.579905. The **0.46%** relative gain versus v1 has an interval of **-0.03% to
1.01%**. Compared with the interaction model, the gain is **0.11%**, with an
interval of **-0.15% to 0.37%**. Thus the former mean regression is repaired,
while an additional K5 predictive advantage is unresolved.

## What the learned mixing does

For ecological affine memory, K0 uses gamma = 0, 0.25, 0.5 and 1 in one, two,
four and two packages. At K5, the source-validation choices contract to
**gamma = 0 in three packages and gamma = 0.25 in six**. Seven of nine choices
change. The validation results favor a regional prior when local evidence is
scarce and reduce its influence when five local observations can calibrate the
temporal model. This pattern is learned rather than enforced as a monotone
function of K.

All four memory families and both support adapters are retained, including the
global controls and every fixed-v1 reference. No target-query result chooses
the deployed mode or mixing weight. These same-cohort station partitions have
been used during model development; this remains developmental evidence.

## Remaining performance problem

High-DOC reconstruction is still the main unresolved error. At K5, Q90 MAE is
6.680758 mg/L versus 6.674577 for the interaction model. Their difference is
0.006181 mg/L, with an interval of -0.002134 to 0.016400. The v2 gain over the
fixed prior is also unresolved in the tail. K0 Q90 MAE remains 7.431761 mg/L.
Ordinary-concentration improvements and useful regional calibration do not yet
solve high-DOC underprediction.

The next substantive model change should address how source stations represent
high-DOC regimes and how the native-scale residual branch learns their errors,
rather than expanding the gamma search. The current iteration provides the
support-aware integration needed to retain zero-observation spatial gains
without carrying the full regional correction into well-supported stations.

## Reproduction and products

Nine packages completed, containing 72 validation-selected wrappers. Recorded
fitting/product time totals **14.0 seconds** using cached experts, excluding
earlier forest/neural training, analysis and replay. This iteration did not
retrain the neural backbone.

All 72 complete wrapper states reproduce exactly from validation episodes.
There are 288 exact K-dependent full-grid base replays and 360 exact model-by-K
query replays. All 2,101,302 cached component-grid rows are unchanged from v1.
K0 equals v1 bitwise; gamma zero equals the unchanged interaction adapter.

The full test suite passes: **631 passed, 2 skipped**. Ruff passes and the
historical artifact audit exits zero. Analysis uses 5,000 whole-station
bootstrap draws and reports all 18 current/fixed models. Detailed product
roles are in [PRODUCTS.md](PRODUCTS.md): the saved full grid is a frozen v1
component cache; new K-dependent query predictions are in predictions.parquet.

```bash
uv run python scripts/run_ladder.py --experiment doc-ecological-transfer-v2
uv run python scripts/analyze_doc_ecological_transfer_v2.py
uv run python scripts/verify_doc_ecological_transfer_v2.py
```

[Detailed results](analysis/findings.md) · [Replay](verification/replay_checks.json)
