# Four DOC residual objectives: completed comparison

Changing the regression objective improves ordinary-concentration reconstruction
in some products, but does not remove the tension between false-high predictions
and high-DOC recovery. The same encoder, observation-aware GRU and concentration
head were fitted with four source objectives: tail-weighted MAE, ordinary MAE,
an ordinary-overprediction penalty, and equal total weight per source station.

## Results at the initial common budget

Each objective has nine fits: three station partitions and three seeds. The
60-epoch ceiling and patience5 give 36 fits and 1,360 executed epochs. The
ecological memory and station-support basis remain fixed; their mixing and
adaptation parameters are selected independently from source validation.

| Integrated model, GRU support basis | K0 MAE | K5 MAE |
|---|---:|---:|
| Previous overall ecological-affine reference | 1.809217 | 1.579905 |
| Tail-weighted control | 1.810792 | 1.588681 |
| Ordinary MAE | 1.808602 | 1.576456 |
| Ordinary-overprediction penalty | 1.813604 | 1.574045 |
| Equal source-station weight | 1.825026 | 1.604613 |

Errors are in mg/L, averaging seeds within partition and then partitions
equally. The same fixed queries and retrospective K-shot support are used.
The complete saved curves, rather than this rounded summary, define the
results; see [analysis](analysis/findings.md).

The ordinary-MAE and penalty arms lower ordinary error and false-high rates
relative to the tail-weighted control at K0, while worsening high-DOC error
and recall. Equal station weighting does not improve the existing pooled
reconstruction endpoint. Source sampling concentration alone does not show
that reweighting will improve transfer.

## Why the common budget was extended

Source-validation traces, inspected before the new target comparisons, showed
that some alternative objectives were still improving at epoch60. The
[separate 120-epoch experiment](../doc_selective_residual_v2/completion.md)
retains every setting, including patience5, and completes that optimization
comparison. This root remains the record of the original duration. No loss
coefficient or architecture was changed using the target results.

## Reproduction

All 36 checkpoints and 864 model-by-K query panels independently replay. Neural
predictions match bitwise; forest recomputation differs by at most 4.97e-14.
The tail-weighted control reproduces the prior encoder model, including its
training trace and direct/integrated predictions.

```bash
uv run python scripts/run_ladder.py --experiment doc-selective-residual-v1
uv run python scripts/analyze_doc_selective_residual_v1.py \
  --root experiments/phase4_transfer/doc_selective_residual_v1
uv run python scripts/verify_doc_selective_residual_v1.py \
  --root experiments/phase4_transfer/doc_selective_residual_v1
```

These are same-cohort development experiments. Whole-station paired intervals,
partition/seed directions, Q90 recall, false-high rates and every failed arm
are retained in the analysis. [Source-validation diagnosis](diagnostics/loss_validation_diagnostic.md)
and [product guide](PRODUCTS.md) explain the training and prediction roles.
