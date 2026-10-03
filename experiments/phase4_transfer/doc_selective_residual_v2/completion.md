# Converged DOC loss comparison: results and next direction

The ordinary-overprediction penalty produces a better support-adapted candidate,
but does not yet improve reconstruction at an entirely unobserved station.
The experiment identifies an objective tradeoff: a lower false-high rate can
come with weaker high-DOC recovery. Equal source-station weighting does not
solve the spatial-transfer problem under the existing pooled endpoint.

## Common model and converged optimization

The existing final self layer, ecological encoder and observation-aware GRU
are updated from the same original expert. The native residual head conditions
on ecology, hydrology and predicted concentration. Four source objectives
change only error costs or station weights; uniform cell shuffling, source OOF
bases, validation selection and inference are unchanged. The ecological memory
and support basis are fixed.

This root contains 36 fits, complementing the preceding 36-fit, 60-epoch root.
The 120-epoch ceiling was set from source-validation cap flags before reading
the new target comparisons. All fits now exhaust patience, stopping by epoch74;
the latest selected checkpoint is epoch69. Total executed epochs are 1,402.
All 36 earlier traces match their extended prefixes exactly. The tail-weighted
control is completely unchanged; no further duration extension is indicated.

## Spatial reconstruction

| Integrated model, GRU support basis | K0 MAE | K1 MAE | K3 MAE | K5 MAE |
|---|---:|---:|---:|---:|
| Previous overall ecological-affine reference | 1.809217 | 1.791696 | 1.622073 | 1.579905 |
| Tail-weighted control | 1.810792 | 1.7769 | 1.6254 | 1.588681 |
| Ordinary MAE | 1.808591 | 1.7836 | 1.6301 | 1.576756 |
| Ordinary-overprediction penalty | 1.813938 | 1.7760 | 1.6124 | **1.572667** |
| Equal source-station weight | 1.824477 | 1.7931 | 1.6203 | 1.604281 |

Errors are in mg/L. Seed means are averaged within each of three partitions,
then partitions receive equal weight. K specifies the number of retrospective
target-station observations available to the existing adapter. Query cells
remain fixed for all K values and models.

At K5, the integrated penalty arm reduces MAE by **1.008%** versus the matched
tail-weighted control, with a paired 95% gain interval of **0.025–2.093%**.
All three partition means improve. The gain is mainly ordinary-concentration
error: ordinary MAE falls from 1.0041 to 0.9827 mg/L; the Q90 error difference
has an interval spanning zero.

Against the previous overall reference, K5 MAE falls from 1.579905 to 1.572667,
a **0.458%** point improvement, interval **-0.121–1.148%**. All three partition
means and seven of nine fits improve, but the broader station interval spans
zero. K0 rises from 1.809217 to 1.813938. The penalty model is retained as a
support-adapted candidate; the previous model remains the overall reference.

Equal station weighting worsens integrated K5 MAE by 1.543% versus the previous
reference, with a gain interval entirely below zero. It is not carried forward
as a performance upgrade.

## Ordinary error and high-DOC recovery

At K0, relative to the matched tail-weighted integrated model:

- Ordinary MAE falls by 0.0321 mg/L with ordinary MAE training and 0.0220 with
  the overprediction penalty.
- Q90 MAE rises by 0.2620 and 0.2275 mg/L, respectively; both intervals exclude
  zero.
- Q90 recall falls by 3.39 and 1.64 percentage points; both intervals exclude
  zero. False-high rates also fall.

Thus the false-alarm reduction is partly a sensitivity tradeoff. At K5, the
penalty candidate is closer to the old overall model's tail performance:
Q90 MAE is 6.6872 versus 6.6808, and recall is 64.45% versus 64.26% (difference
interval spans zero). Its false-high rate is 2.183% versus 2.135%, a small
increase of 0.048 percentage points, interval 0.012–0.102. Better ordinary MAE
does not imply fewer threshold errors relative to every baseline.

The remaining task is to identify when a new station or month truly needs a
large DOC correction. Source-only input analysis points to within-month
hydrologic variation, which the current monthly-mean discharge input discards,
as a concrete next information experiment. Further loss-weight searches or
automatic epoch increases are not the next model change.

## Reproduction and completion

The independent replay covers nine packages, 36 checkpoints, 2,101,302
full-grid rows and 864 model-by-K panels (3,705,120 query rows). Neural and
adapted predictions reproduce bitwise. Source-validation adapter/mixer refits,
tail-weighted control traces and weights, and station weight totals reproduce.
Forest recomputation differs by at most 4.97e-14.

The full test suite passes **684 tests, 2 skipped**. Ruff and the historical
artifact audit pass. Older experiments and endpoints remain unchanged. These
are previously examined development partitions; all loss arms and comparisons
are reported using 5,000 joint whole-station bootstrap draws.

```bash
uv run python scripts/run_ladder.py --experiment doc-selective-budget-v1
uv run python scripts/analyze_doc_selective_residual_v1.py \
  --root experiments/phase4_transfer/doc_selective_residual_v2 --expected-epochs 120 \
  --budget-reference-root experiments/phase4_transfer/doc_selective_residual_v1
uv run python scripts/verify_doc_selective_residual_v1.py \
  --root experiments/phase4_transfer/doc_selective_residual_v2
```

[Complete comparisons](analysis/findings.md) · [Duration comparison](analysis/budget_extension_findings.md)
· [Source-validation traces](diagnostics/loss_validation_diagnostic.md)
· [Product guide](PRODUCTS.md) · [Independent replay](verification/replay_checks.json)
