# DOC loss comparison: product guide

All four models share the same observation-aware GRU, partially trainable
spatial/ecological encoder and concentration-conditioned native residual head.
Only the training objective differs:

| Saved arm | Training objective |
|---|---|
| `tail2` | Current cell-weighted MAE with twofold Q90 emphasis |
| `mae` | Ordinary cell-weighted MAE |
| `selective` | Tail2 plus a one-sided ordinary-overprediction cost |
| `station` | Equal-station average of within-station tail-weighted MAEs |

Truth-based groups and station loss weights are used only in source training.
Inference takes the same raw input windows and has no extra classifier, gate
or target-query label. Overall unweighted source-validation MAE still selects
the checkpoint and residual scale.

Each run saves four checkpoints, corresponding JSON summaries and training
traces, fixed feature definitions, direct support adapters and ecological
integration wrappers. `full_grid.parquet` contains all357×654 station-months:
context prediction, ecological memory, unscaled neural deltas, scaled/floored
direct predictions and integrated **K0** bases. `predictions.parquet` contains
the final fixed-query predictions at K=0/1/3/5 for all24 reported models.
Support is retrospective. K1/K3/K5 adapted products are not the full-grid
component columns.

Six reference models carry the preceding encoder's direct/integrated predictions
and the earlier overall ecological-affine model unchanged. Tail2 is also
independently matched against that preceding encoder's selected weights,
training trace, predictions and support/mixing choices.

The 60-epoch cap is recorded in this experiment. The duration completion in
[v2](../doc_selective_residual_v2/study_plan.md) retains all objectives but allows
up to120 epochs with the same patience5. It was chosen from source-validation
curves before examining the new target comparisons.

```bash
uv run python scripts/run_ladder.py --experiment doc-selective-residual-v1
uv run python scripts/analyze_doc_selective_residual_v1.py
uv run python scripts/verify_doc_selective_residual_v1.py
uv run python scripts/diagnose_doc_encoder_selectivity_v1.py
```
