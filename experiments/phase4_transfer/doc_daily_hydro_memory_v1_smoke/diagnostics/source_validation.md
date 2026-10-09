# Does daily hydrology improve the recurrent state?

Partial execution diagnostic; formal nine-run evidence is incomplete.
Only source-validation queries, training traces and saved validation choices are used here. The validation set also selected checkpoints and calibration coefficients; these are development diagnostics, not independent transfer estimates.

The off control retains daily descriptors at the scalar head. Current-only adds their zero-initialized projection at the final GRU step; full-history adds it at every valid causal step. The two active modes add the same 512 parameters at hidden size 64. Tree probes use the same source labels and daily-history slots with the preceding forest settings; their predictions do not replace the neural model's fixed context base.

## Training

| Arm | Validation MAE | Selected epochs | Epochs run | Scale counts | Projection norm | Cap-limited |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| off | 2.098116 | 3–3 | 3–3 | 1.0: 1 | 0.00000–0.00000 | 1/1 |
| current_only | 2.099208 | 3–3 | 3–3 | 1.0: 1 | 0.13755–0.13755 | 1/1 |
| full_history | 2.099037 | 3–3 | 3–3 | 1.0: 1 | 0.13454–0.13454 | 1/1 |

Off-control traces are exactly equal to the preceding daily-head model in 0/1 packages. The corresponding fixed validation predictions are equal in 0/1. All recorded losses are finite. Recorded package durations total 34.5 seconds, including fitting, inference and saved products.

## K0 errors by daily information

| Stage | Group | Off MAE | Current-only MAE | Full-history MAE | Full−current | Full-history bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| direct | all | 2.098116 | 2.099208 | 2.099037 | -0.000170 | -0.702203 |
| direct | none_valid | 1.480881 | 1.480071 | 1.479642 | -0.000429 | -0.372511 |
| direct | some_valid | 2.189258 | 2.190631 | 2.190499 | -0.000132 | -0.750886 |
| direct | history_0 | 1.499103 | 1.498221 | 1.497613 | -0.000607 | -0.382666 |
| direct | history_1_5 | 2.212638 | 2.209397 | 2.210532 | +0.001134 | -1.283263 |
| direct | history_6_11 | 2.719377 | 2.720742 | 2.723284 | +0.002541 | -0.977729 |
| direct | history_12 | 2.153637 | 2.155170 | 2.154877 | -0.000293 | -0.716838 |
| integrated | all | 2.098116 | 2.099116 | 2.098652 | -0.000464 | -0.724404 |
| integrated | none_valid | 1.480881 | 1.477476 | 1.477056 | -0.000420 | -0.349202 |
| integrated | some_valid | 2.189258 | 2.190909 | 2.190438 | -0.000471 | -0.779807 |
| integrated | history_0 | 1.499103 | 1.495099 | 1.494742 | -0.000357 | -0.361130 |
| integrated | history_1_5 | 2.212638 | 2.226653 | 2.227503 | +0.000851 | -1.302443 |
| integrated | history_6_11 | 2.719377 | 2.697847 | 2.699178 | +0.001331 | -1.031186 |
| integrated | history_12 | 2.153637 | 2.156174 | 2.155551 | -0.000623 | -0.744244 |

Native MAE and signed bias are in mg/L; positive bias means overprediction. Daily validity comes from the frozen data product. History groups count months with at least one valid descriptor in the causal 12-month window, including the current month. Sparse or empty groups are descriptive; their counts are saved by partition. Each displayed result first averages seeds within partition, then gives available partitions equal weight.

## Partition directions

| Stage | Partition | Current−off MAE | Full−off MAE | Full−current MAE | Full improves current seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 142 | +0.001092 | +0.000922 | -0.000170 | 1/1 |
| integrated | 142 | +0.001000 | +0.000536 | -0.000464 | 1/1 |

## Selected validation K curves

| Model | K=0 | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: |
| context_gru_tuned_anchor | 2.146987 | 2.037445 | 1.854607 | 1.778317 |
| off_gru_tuned_anchor | 2.098116 | 1.999619 | 1.850176 | 1.761219 |
| current_only_gru_tuned_anchor | 2.099208 | 2.000034 | 1.850669 | 1.761725 |
| full_history_gru_tuned_anchor | 2.099037 | 1.999336 | 1.850615 | 1.761594 |
| tree_current_gru_tuned_anchor | 2.124283 | 2.030390 | 1.862367 | 1.775575 |
| tree_history_gru_tuned_anchor | 2.127621 | 2.012662 | 1.856178 | 1.774520 |
| off_integrated_gru_tuned_anchor | 2.098116 | 1.999619 | 1.845048 | 1.758858 |
| current_only_integrated_gru_tuned_anchor | 2.099116 | 1.999879 | 1.845105 | 1.758936 |
| full_history_integrated_gru_tuned_anchor | 2.098652 | 1.999150 | 1.845054 | 1.758587 |

Both constant and temporal support-adapter selections are preserved in the CSV; the table shows the existing temporal basis. Tree-history versus tree-current asks whether these same historical covariates help an explicit-feature model. A neural history gain alone does not establish a new river-transport mechanism.

## Integrated K5 choices

| Mode | Gamma counts | Alpha counts | Ridge counts |
| --- | --- | --- | --- |
| off | 0.25: 1 | 0.75: 1 | 1.0: 1 |
| current_only | 0.25: 1 | 0.75: 1 | 1.0: 1 |
| full_history | 0.25: 1 | 0.75: 1 | 1.0: 1 |

## What this recurrent state represents

The existing DOC-age/support decay acts on the complete previous hidden state, including hydrologic information. At a station with no visible DOC, the original age feature increases with elapsed calendar months rather than with the age of discharge measurements. The experiment therefore tests daily hydrology within the existing observation-conditioned recurrence; it does not create a separately timed hydrologic store. A nonzero projection norm establishes that the branch was trained, not that its weights are physical transport coefficients.

No feature, route, epoch budget or model is changed by this diagnostic. Spatial transfer and high-DOC tradeoffs are evaluated separately on the retained target outputs.
