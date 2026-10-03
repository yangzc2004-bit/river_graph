# Matched 60-epoch budget: product guide

The model and product layout are described in the preceding
[encoder-tuning guide](../doc_encoder_residual_v1/PRODUCTS.md). This experiment
changes only the maximum epoch count from 30 to 60, retaining patience 5 and
starting every fit from its original weights. Early stopping can still select
an earlier checkpoint.

Per-run configurations retain the implementation identifier
`doc_encoder_residual_v1` and explicitly record `epochs: 60`. The separate
directory, [duration plan](study_plan.md), execution wrapper and runtime
snapshot identify this budget extension. Old references remain 30-epoch
models; their differences from the new frozen control include duration.

`full_grid.parquet` holds neural component predictions and integrated K0
bases. Final K1/K3/K5 adapted query outputs are in `predictions.parquet`.
Source-validation refits reproduce the checkpoint scales, support adapters
and mixing choices; outer query truth is only used in the analysis.

```bash
uv run python scripts/run_ladder.py --experiment doc-encoder-budget-v1
uv run python scripts/analyze_doc_encoder_residual_v1.py \
  --root experiments/phase4_transfer/doc_encoder_budget_v1 --expected-epochs 60
uv run python scripts/verify_doc_encoder_residual_v1.py \
  --root experiments/phase4_transfer/doc_encoder_budget_v1
uv run python scripts/verify_doc_encoder_budget_v1.py
```

The paired duration analysis compares each mode with its own saved 30-epoch
counterpart. The additional budget replay checks the exact initial trace prefix
and unchanged checkpoints for fits that already exhausted patience at 30.
The source-validation diagnostic reports how much of the validation benefit is
shared by the frozen control versus added encoder adaptation.
