# Converged DOC loss comparison: product guide

Product meanings and the four loss definitions are documented in
[the preceding guide](../doc_selective_residual_v1/PRODUCTS.md). This root changes
only the maximum epoch budget from60 to120, retaining patience5 and original
initialization. Every fit may stop sooner.

The implementation identifier remains `doc_selective_residual_v1`; per-run
configurations explicitly record `epochs:120`. The new directory, wrapper,
study plan and runtime snapshot identify the duration completion. Source
validation determines checkpoints, residual scales and support/mixing choices.
It does not change the loss definitions or primary target-comparison metric.

`full_grid.parquet` contains neural components and integrated K0 bases;
`predictions.parquet` contains final K-dependent query predictions. The fixed
query set and retrospective support are unchanged. The prior tail2 control
already exhausted patience below60, so its selected weights and predictions
must stay unchanged under the larger maximum.

```bash
uv run python scripts/run_ladder.py --experiment doc-selective-budget-v1
uv run python scripts/analyze_doc_selective_residual_v1.py \
  --root experiments/phase4_transfer/doc_selective_residual_v2 --expected-epochs 120
uv run python scripts/verify_doc_selective_residual_v1.py \
  --root experiments/phase4_transfer/doc_selective_residual_v2
```
