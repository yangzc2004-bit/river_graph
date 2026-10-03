# Encoder-tuning product guide

This experiment updates the existing DOC recurrent residual using raw spatial
inputs. It compares frozen encoding, final self-layer tuning, and final
self-layer plus ecological-encoder tuning. All modes use the same
concentration-conditioned head and source-validation selection.

## Saved products

Each `runs/split<partition>_seed<seed>/` package contains:

- `frozen.pt`, `last_self.pt`, `last_self_ecology.pt`: initial and selected
  spatial/GRU/decay/head states, architecture, training settings and selection
  summary. These checkpoints reconstruct the model without neural refitting.
- Matching JSON summaries and epoch trace CSVs: source-validation decisions
  and parameter changes, including the exact frozen-encoder control.
- `feature_definition.json`: the source-fitted extra-feature normalization and
  concentration-interaction layout. Historical expert input normalization is
  reused unchanged.
- `adapters.json`: validation-selected direct station-support adapters.
- `mixers.json`: validation-selected ecological-memory integration and support
  adapters. Mixing choices may differ with K.
- `full_grid.parquet`: all 357 × 654 station-months, containing the context
  base, unchanged ecological memory, each neural delta and direct prediction,
  and each integrated **K0** prediction. It is a component product; it does
  not contain the adapted K1/K3/K5 final predictions.
- `predictions.parquet`: fixed query cells for all 18 reported models and
  K=0/1/3/5. `y_pred` is the final adapted prediction; `base_pred` is its
  selected pre-adaptation base. Query truth is appended after inference.
- Prediction sidecars, runtime snapshot and `complete.json`: saved execution
  settings and the source packages needed to reproduce the outputs.

Both concentration and previous overall ecological-affine references are
carried forward without changing their predictions. Reserved station support
is retrospective: a support month may occur after a query month. The recurrent
input window itself only reads the current and preceding months.

## Model scope

The spatial encoder uses empty edges in every arm, matching the current
spatial-transfer residual's self path. This experiment tests representation
adaptation; it does not add upstream transport information. The existing river
message tensors and old spatial prediction head remain frozen.

## Reproduction

```bash
uv run python scripts/run_ladder.py --experiment doc-encoder-residual-v1
uv run python scripts/analyze_doc_encoder_residual_v1.py
uv run python scripts/verify_doc_encoder_residual_v1.py
```

Independent replay checks raw versus cached initial representations separately
from trained model outputs. Float32 initialization comparisons use numerical
tolerance; saved-model inference reports its actual equality and maximum
difference. A small initial numerical difference is not used to demand equal
optimization trajectories between the old cached experiment and the new raw
control.
