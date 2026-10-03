# DOC encoder training duration: completed comparison

Extending the matched encoder experiment to a 60-epoch ceiling improves some
zero-observation predictions, but does not establish a further overall gain.
The frozen encoder catches up with part of the advantage seen under the
30-epoch budget. All fits stop before 60; another duration increase is not the
next useful model change.

## Matched spatial reconstruction

| Direct model with the same GRU support adapter | K0 MAE, 30 cap | K0 MAE, 60 cap | K5 MAE, 30 cap | K5 MAE, 60 cap |
|---|---:|---:|---:|---:|
| Frozen concentration encoder | 1.843635 | 1.831279 | 1.588311 | 1.593497 |
| Final self-layer update | 1.835723 | 1.827988 | 1.588283 | 1.593607 |
| Final self-layer + ecology update | 1.829927 | 1.823984 | 1.586527 | 1.594694 |

Errors are in mg/L, with seeds averaged within each partition and partitions
equally weighted. The same fixed queries and retrospective K-shot support are
used in both experiments.

Longer training reduces frozen K0 MAE by 0.670%, interval -0.316% to 1.623%.
The combined encoder update gains another 0.325% over its own 30-epoch product,
interval -0.618% to 1.179%. Neither establishes a duration benefit across the
target station population. At the matched 60-epoch budget, combined updating
improves K0 by 0.398% over frozen, interval -0.407% to 1.257%; two partition
means and seven of nine fits improve. The stronger 30-epoch contrast does not
persist at the same strength once the control receives more optimization.

K5 mean error increases for every direct mode. Their paired duration intervals
span zero. More accurate K0 predictions do not automatically provide a better
base for the existing station-support adapter.

## Integration and high-DOC performance

| Ecological-integrated product | K0 MAE | K5 MAE |
|---|---:|---:|
| Earlier overall ecological-affine model | **1.809217** | **1.579905** |
| Combined encoder update, 30 cap | 1.813615 | 1.584405 |
| Combined encoder update, 60 cap | 1.810792 | 1.588681 |

The previous overall model remains the performance reference. The new
integrated K0 mean comes close, but does not improve it; K5 is also worse on
average. Paired intervals span zero.

At K0, the 60-cap integrated candidate reduces Q90 MAE from 7.431761 to
**7.251297 mg/L** relative to the earlier overall model: **2.428%**, interval
**1.010% to 3.452%**, with all three partition means improving. Its non-tail
MAE rises by 0.022405 mg/L and false-high rate rises by 0.193 percentage
points, interval 0.096 to 0.326. A K5 tail benefit is no longer established.
The remaining performance problem is selective correction: improving high
values without raising ordinary predictions.

## What the source-validation traces show

Fourteen of 27 fits select a checkpoint after epoch 30. The latest selected
checkpoint is epoch 50, and all fits stop by epoch 55. Total executed epochs
increase from 645 to 862. Patience, learning rates, initialization, objective,
tail weighting, queries and adapter grids are unchanged.

Mean additional source-validation MAE reduction is 0.01711 mg/L for frozen,
0.01382 for final-self tuning and 0.02352 for combined tuning. Partition 142
largely shows a shared optimization benefit. The larger ecological average is
concentrated in one delayed-improvement fit in partition 143; partition 144
changes little. These patterns support completing the duration comparison,
but do not support a higher epoch ceiling or broad claims about encoder depth.

## Completion and reproduction

Nine packages contain 27 fits (297.2 seconds of recorded fitting/product time)
and all 18 reported models. Analysis retains the
45 encoder contrasts and adds 24 separate paired budget contrasts, using 5,000
joint whole-station bootstrap draws. The same previously examined cohort and
source validation remain development data. Old models and references are
preserved.

All 27 checkpoints and 648 model-by-K panels reproduce bitwise from saved
weights. The 672 old trace rows, including epoch zero, exactly match their
new prefixes. Thirteen already stopped models retain identical selected
weights and predictions; the other fourteen account for all 217 additional
epochs. Independent source-validation adapter/mixer refits also match.
Forest recomputation differs by at most 4.97e-14. The full suite passes
**667 tests, 2 skipped**; Ruff and the historical artifact audit pass.

```bash
uv run python scripts/run_ladder.py --experiment doc-encoder-budget-v1
uv run python scripts/analyze_doc_encoder_residual_v1.py \
  --root experiments/phase4_transfer/doc_encoder_budget_v1 --expected-epochs 60 \
  --budget-reference-root experiments/phase4_transfer/doc_encoder_residual_v1
uv run python scripts/verify_doc_encoder_residual_v1.py \
  --root experiments/phase4_transfer/doc_encoder_budget_v1
uv run python scripts/verify_doc_encoder_budget_v1.py
```

[Full results](analysis/findings.md) · [Paired duration analysis](analysis/budget_extension_findings.md)
· [Source-validation diagnostic](diagnostics/source_validation_budget_diagnostic.md)
· [Product guide](PRODUCTS.md) · [Replay](verification/replay_checks.json)
· [Duration continuity](verification/budget_continuation_checks.json)
