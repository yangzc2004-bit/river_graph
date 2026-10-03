# Does daily hydrology improve the recurrent state?

All nine formal packages are complete.
Only source-validation queries, training traces and saved validation choices are used here. The validation set also selected checkpoints and calibration coefficients; these are development diagnostics, not independent transfer estimates.

The off control retains daily descriptors at the scalar head. Current-only adds their zero-initialized projection at the final GRU step; full-history adds it at every valid causal step. The two active modes add the same 512 parameters at hidden size 64. Tree probes use the same source labels and daily-history slots with the preceding forest settings; their predictions do not replace the neural model's fixed context base.

## Training

| Arm | Validation MAE | Selected epochs | Epochs run | Scale counts | Projection norm | Cap-limited |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| off | 1.804428 | 27–72 | 32–77 | 1.0: 9 | 0.00000–0.00000 | 0/9 |
| current_only | 1.801147 | 31–57 | 36–62 | 1.0: 9 | 0.24576–0.59122 | 0/9 |
| full_history | 1.798270 | 31–67 | 36–72 | 1.0: 9 | 0.24735–0.48078 | 0/9 |

Off-control traces are exactly equal to the preceding daily-head model in 9/9 packages. The corresponding fixed validation predictions are equal in 9/9. All recorded losses are finite. Recorded package durations total 704.6 seconds, including fitting, inference and saved products.

## K0 errors by daily information

| Stage | Group | Off MAE | Current-only MAE | Full-history MAE | Full−current | Full-history bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| direct | all | 1.804428 | 1.801147 | 1.798270 | -0.002876 | -0.651104 |
| direct | none_valid | 1.373486 | 1.375289 | 1.386106 | +0.010818 | -0.538164 |
| direct | some_valid | 1.931638 | 1.927014 | 1.921316 | -0.005698 | -0.682063 |
| direct | history_0 | 1.372973 | 1.376035 | 1.386677 | +0.010642 | -0.525582 |
| direct | history_1_5 | 2.067521 | 2.058558 | 2.032631 | -0.025928 | -0.378666 |
| direct | history_6_11 | 2.337165 | 2.338239 | 2.315988 | -0.022251 | -0.265141 |
| direct | history_12 | 1.906120 | 1.901129 | 1.896908 | -0.004221 | -0.704030 |
| integrated | all | 1.804067 | 1.800971 | 1.797794 | -0.003177 | -0.668038 |
| integrated | none_valid | 1.374141 | 1.375171 | 1.387107 | +0.011935 | -0.552005 |
| integrated | some_valid | 1.930753 | 1.926808 | 1.920077 | -0.006732 | -0.700596 |
| integrated | history_0 | 1.373709 | 1.375943 | 1.387855 | +0.011912 | -0.539545 |
| integrated | history_1_5 | 2.063655 | 2.058233 | 2.024920 | -0.033313 | -0.370097 |
| integrated | history_6_11 | 2.330691 | 2.335074 | 2.306385 | -0.028689 | -0.271650 |
| integrated | history_12 | 1.905516 | 1.901037 | 1.896115 | -0.004922 | -0.723710 |

Native MAE and signed bias are in mg/L; positive bias means overprediction. Daily validity comes from the frozen data product. History groups count months with at least one valid descriptor in the causal 12-month window, including the current month. Sparse or empty groups are descriptive; their counts are saved by partition. Each displayed result first averages seeds within partition, then gives available partitions equal weight.

## Partition directions

| Stage | Partition | Current−off MAE | Full−off MAE | Full−current MAE | Full improves current seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 142 | -0.001591 | +0.014844 | +0.016434 | 0/3 |
| direct | 143 | -0.008132 | -0.036559 | -0.028427 | 3/3 |
| direct | 144 | -0.000121 | +0.003243 | +0.003363 | 0/3 |
| integrated | 142 | -0.001591 | +0.014844 | +0.016434 | 0/3 |
| integrated | 143 | -0.008132 | -0.036559 | -0.028427 | 3/3 |
| integrated | 144 | +0.000434 | +0.002896 | +0.002462 | 0/3 |

## Selected validation K curves

| Model | K=0 | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: |
| context_gru_tuned_anchor | 1.990984 | 1.922744 | 1.711944 | 1.643696 |
| off_gru_tuned_anchor | 1.804428 | 1.774421 | 1.675127 | 1.616446 |
| current_only_gru_tuned_anchor | 1.801147 | 1.772615 | 1.673770 | 1.614725 |
| full_history_gru_tuned_anchor | 1.798270 | 1.768582 | 1.672602 | 1.614508 |
| tree_current_gru_tuned_anchor | 1.944702 | 1.885570 | 1.696589 | 1.631860 |
| tree_history_gru_tuned_anchor | 1.940824 | 1.884477 | 1.701123 | 1.635427 |
| off_integrated_gru_tuned_anchor | 1.804067 | 1.774060 | 1.663327 | 1.604602 |
| current_only_integrated_gru_tuned_anchor | 1.800971 | 1.772440 | 1.663218 | 1.604251 |
| full_history_integrated_gru_tuned_anchor | 1.797794 | 1.768107 | 1.663310 | 1.604384 |

Both constant and temporal support-adapter selections are preserved in the CSV; the table shows the existing temporal basis. Tree-history versus tree-current asks whether these same historical covariates help an explicit-feature model. A neural history gain alone does not establish a new river-transport mechanism.

## Integrated K5 choices

| Mode | Gamma counts | Alpha counts | Ridge counts |
| --- | --- | --- | --- |
| off | 0.0: 1; 0.25: 4; 0.5: 4 | 0.5: 3; 0.75: 3; 1.0: 3 | 1.0: 9 |
| current_only | 0.0: 2; 0.25: 2; 0.5: 5 | 0.5: 3; 0.75: 3; 1.0: 3 | 1.0: 8; 10.0: 1 |
| full_history | 0.0: 1; 0.25: 4; 0.5: 4 | 0.5: 3; 0.75: 3; 1.0: 3 | 1.0: 7; 10.0: 2 |

## What this recurrent state represents

The existing DOC-age/support decay acts on the complete previous hidden state, including hydrologic information. At a station with no visible DOC, the original age feature increases with elapsed calendar months rather than with the age of discharge measurements. The experiment therefore tests daily hydrology within the existing observation-conditioned recurrence; it does not create a separately timed hydrologic store. A nonzero projection norm establishes that the branch was trained, not that its weights are physical transport coefficients.

No feature, route, epoch budget or model is changed by this diagnostic. Spatial transfer and high-DOC tradeoffs are evaluated separately on the retained target outputs.
