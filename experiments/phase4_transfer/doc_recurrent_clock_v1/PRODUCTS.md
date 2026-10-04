# Recurrent-clock products

Each nine-package run contains a copied `legacy.pt` / `legacy.json` from the
verified daily-head parent and new `unseen_neutral` / `flow_window` checkpoints,
summaries and epoch traces. The new versions update the same permitted
encoder/recurrent/head parameters; only the scalar used by decay is redefined.
Raw M1 DOC age remains unchanged. Legacy allocation and training-budget
controls are preserved; neutral changes effective decay capacity.

`feature_definition.json` is copied unchanged. Source native residual training
uses station OOF forest predictions with held-station-fold-hidden raw DOC
inputs, current38-channel features and the same Q90-weighted native loss.
Validation/test predictions use the frozen full-source forest. The support
basis and ecological residual profile are inherited, not refitted here.

`full_grid.parquet` covers233,478 station-months and contains context,
ecological memory, each clock's native residual/prediction and integrated K0
base. Availability metadata is included; raw daily-feature columns are in the
unchanged bound daily pack, not duplicated in this full grid.
`predictions.parquet` holds14 model/basis choices ×K0/1/3/5 on identical
target queries, with labels appended after training and calibration.

`adapters.json` / `mixers.json` contain source-validation support and mixture
choices; `source_validation.csv` contains48 direct/integrated model/K panels.
`clock_diagnostics.json` contains label-free validation-window clock, gamma and
explicit multiplicative retention summaries. Retention is not the complete
GRU Jacobian or a physical transmission coefficient. Padding is excluded.

The analysis reports all eight fixed clock-versus-legacy contrasts, full K
curves, ordinary/Q90 errors, detection consequences, station/partition
consistency and source selection. Its separate decay-capacity diagnostic
reconstructs every source-validation query window without target labels.
Independent verification checks visibility/feature reconstruction, hand-derived
clocks and gamma, actual recurrence, native predictions, adaptation and exact
legacy controls. Sidecars and completion manifests bind each product.

The executed archive remains immutable; a post-run import-layout edit has an
identical AST and is recorded separately. `_smoke/` is excluded from
scientific aggregation. No new model is promoted by these results.
