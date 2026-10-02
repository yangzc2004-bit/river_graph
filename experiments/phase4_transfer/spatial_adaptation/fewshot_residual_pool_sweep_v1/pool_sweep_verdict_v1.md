# Source-pool size sweep for the K=5 spatial residual adapter

This development extension tests whether the hard source-station pool size
K-source = 40 limits the existing K=5 support-residual adapter. The candidate
pool sizes are 20, 40, 80 and 160 source stations. For each pool, alpha is
selected on the internal E3 station-held-out validation split. The outer E3
split is scored only for the selected pair and the existing K-source = 40,
alpha = 0.75 mean-residual control.

The experiment uses three seeds (42--44) and 300-tree ExtraTrees models. The
query set remains the same 2,316 outer E3 cells and target support is hidden
when source models generate query and support predictions.

## Result

Internal validation selected:

* source pool size: **40**;
* residual alpha: **0.75**;
* validation MAE: **1.7774**.

The outer selected result is MAE **2.0239 ± 0.0054** across the three seeds.
Because the selected pair is exactly the existing K-source = 40,
alpha = 0.75 control, the two outer reports are identical. No alternative
source-pool size was selected or scored as an outer candidate.

The validation ranking was:

| source pool | alpha | validation MAE |
| ---: | ---: | ---: |
| 40 | 0.75 | 1.777 |
| 40 | 0.50 | 1.779 |
| 80 | 0.50 | 1.800 |
| 160 | 0.75 | 1.804 |
| 20 | 0.50 | 1.819 |

The current k=40 choice is therefore retained. This sweep does not support a
new performance claim; it indicates that the present hard source-pool size is
already near the best setting for the K=5 residual adapter under this protocol.

## Reproducible outputs

* Runner: scripts/run_spatial_residual_pool_sweep.py
* Validation candidates: validation_pool_candidates.csv
* Outer selected/control result: test_selected_and_control.csv
* Outer selected/control predictions: test_selected_and_control_predictions.parquet
* Selection and policy manifest: selected_pool.json and manifest.json
