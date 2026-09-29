# R7: Does observation-aware GNN information complement Temporal RF?

Run DOC temporal holdout (`e2a_strict`) and spatial holdout
(`e3_spatial_seed42`) with seeds 42, 43 and 44. Complete all six cases.
This is an exploratory continuation on previously inspected splits.

Fit the current M1 implementation with its baseline two-layer transport
encoder, 12-month GRU-D, hidden width 64, dropout 0.1, mixed point/block/station
masking, learning rate 0.001, 30 epochs and patience 5. Fit Temporal RF with
its existing features, 200 trees and the paired seed. Both fits use the same
train cells. DOC targets use log1p; report errors in mg/L.

Old M1 products have no checkpoints. These are new fits of the current code,
not recovered historical executions. Save both fitted models, validation
predictions and test predictions for subsequent analysis without refitting.

Select alpha_RF from {0, 0.05, ..., 1} by raw-scale validation MAE; break
exact ties toward RF. Predict validation cells with *all* validation and test
labels hidden in both models' inputs (only train/context visible). This same
validation set also selects the GNN checkpoint. Once alpha is saved, predict
test cells using the established train/val/context view. Test labels never
select alpha. The blend is alpha_RF * RF + (1-alpha_RF) * M1.

Report RF, M1 and blend MAE, RMSE, R2, training-Q90 tail MAE (unique n),
per-seed alpha, validation-to-test gain, and paired RF-minus-blend differences.
Average training seeds on each query cell before station-cluster bootstrap
(2,000 draws, seed 42). Positive differences favour the blend. Relative
improvement is 100 * (RF_MAE - blend_MAE) / RF_MAE. Report fixed observation
age and upstream-support strata to locate any complementarity; do not tune
a conditional gate on test strata.

If validation-selected blending consistently helps, next compare against a
matched-input no-message GNN to isolate the river message contribution before
building a dedicated residual branch. If blending selects RF alone or loses,
retain these results and do not expand the residual architecture on this basis.
A blend gain demonstrates complementary predictors, not a graph mechanism.
