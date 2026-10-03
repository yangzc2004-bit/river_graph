# DOC spatial adaptation: updating recurrent states

All nine packages compare the updated GRU with its frozen counterpart under identical
calendar-anchor normalization. Environmental experts, spatial encoders, fixed readouts
and fusion predictions remain unchanged. Partitions 142–144 are previously examined
same-cohort station holdouts, so this is a development comparison.

Negative paired MAE difference and positive relative reduction favor the candidate.
Intervals use 5,000 joint whole-station bootstrap draws. Seed losses are averaged
within partitions, followed by equal partition weighting; repeated station identities
are resampled together. No checkpoint or adapter is selected from these test results.

## Primary: recurrent update with matched normalization

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_tuned_vs_frozen_anchor_k3 | 1.6327 | 1.6348 | 0.0022 [-0.0022, 0.0071] | -0.13% [-0.44, 0.13] | 1/3; 2/9 |
| context_tuned_vs_frozen_anchor_k5 | 1.5981 | 1.5960 | -0.0021 [-0.0060, 0.0015] | 0.13% [-0.09, 0.37] | 2/3; 3/9 |
| fusion_tuned_vs_frozen_anchor_k3 | 1.6279 | 1.6300 | 0.0021 [-0.0024, 0.0070] | -0.13% [-0.43, 0.14] | 1/3; 2/9 |
| fusion_tuned_vs_frozen_anchor_k5 | 1.5973 | 1.5952 | -0.0021 [-0.0060, 0.0016] | 0.13% [-0.10, 0.37] | 2/3; 3/9 |

## Performance against existing adapters

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_tuned_vs_prior_gru_k3 | 1.6395 | 1.6348 | -0.0047 [-0.0130, 0.0045] | 0.29% [-0.26, 0.81] | 3/3; 6/9 |
| context_tuned_vs_tree_k3 | 1.6337 | 1.6348 | 0.0011 [-0.0167, 0.0204] | -0.07% [-1.25, 0.98] | 2/3; 4/9 |
| context_tuned_vs_constant_k3 | 1.6437 | 1.6348 | -0.0089 [-0.0203, 0.0009] | 0.54% [-0.05, 1.19] | 3/3; 8/9 |
| context_tuned_vs_prior_gru_k5 | 1.5995 | 1.5960 | -0.0035 [-0.0081, 0.0009] | 0.22% [-0.05, 0.49] | 3/3; 6/9 |
| context_tuned_vs_tree_k5 | 1.5962 | 1.5960 | -0.0002 [-0.0103, 0.0096] | 0.01% [-0.60, 0.61] | 1/3; 5/9 |
| context_tuned_vs_constant_k5 | 1.6094 | 1.5960 | -0.0134 [-0.0268, -0.0020] | 0.83% [0.12, 1.60] | 3/3; 9/9 |
| fusion_tuned_vs_prior_gru_k3 | 1.6358 | 1.6300 | -0.0058 [-0.0132, 0.0025] | 0.35% [-0.15, 0.82] | 3/3; 7/9 |
| fusion_tuned_vs_tree_k3 | 1.6307 | 1.6300 | -0.0007 [-0.0185, 0.0177] | 0.04% [-1.09, 1.10] | 2/3; 4/9 |
| fusion_tuned_vs_constant_k3 | 1.6392 | 1.6300 | -0.0092 [-0.0219, 0.0010] | 0.56% [-0.07, 1.29] | 3/3; 8/9 |
| fusion_tuned_vs_prior_gru_k5 | 1.5987 | 1.5952 | -0.0035 [-0.0080, 0.0009] | 0.22% [-0.06, 0.49] | 3/3; 6/9 |
| fusion_tuned_vs_tree_k5 | 1.5952 | 1.5952 | 0.0000 [-0.0100, 0.0098] | -0.00% [-0.62, 0.60] | 1/3; 5/9 |
| fusion_tuned_vs_constant_k5 | 1.6085 | 1.5952 | -0.0133 [-0.0267, -0.0019] | 0.83% [0.12, 1.60] | 3/3; 9/9 |

## Normalization change with recurrence frozen

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_frozen_anchor_vs_prior_gru_k3 | 1.6395 | 1.6327 | -0.0068 [-0.0157, 0.0020] | 0.42% [-0.12, 0.97] | 2/3; 5/9 |
| context_frozen_anchor_vs_prior_gru_k5 | 1.5995 | 1.5981 | -0.0014 [-0.0056, 0.0025] | 0.09% [-0.15, 0.34] | 1/3; 6/9 |
| fusion_frozen_anchor_vs_prior_gru_k3 | 1.6358 | 1.6279 | -0.0078 [-0.0160, -0.0004] | 0.48% [0.02, 0.99] | 2/3; 6/9 |
| fusion_frozen_anchor_vs_prior_gru_k5 | 1.5987 | 1.5973 | -0.0014 [-0.0056, 0.0025] | 0.09% [-0.15, 0.34] | 1/3; 6/9 |

## Error profiles

| Predictor | K | MAE | RMSE | R² | Log MAE | Training-Q90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Environmental + constant correction | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + constant correction | 5 | 1.6094 | 3.6146 | 0.5621 | 0.2323 | 6.7436 |
| Environmental + prior GRU projection | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + prior GRU projection | 5 | 1.5995 | 3.5993 | 0.5657 | 0.2297 | 6.6969 |
| Environmental + frozen GRU, anchor normalized | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + frozen GRU, anchor normalized | 5 | 1.5981 | 3.5969 | 0.5663 | 0.2296 | 6.7002 |
| Environmental + updated GRU, anchor normalized | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + updated GRU, anchor normalized | 5 | 1.5960 | 3.5911 | 0.5675 | 0.2294 | 6.6945 |
| Environmental + prior tree projection | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + prior tree projection | 5 | 1.5962 | 3.6027 | 0.5652 | 0.2288 | 6.7182 |
| Frozen fusion + constant correction | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + constant correction | 5 | 1.6085 | 3.6113 | 0.5627 | 0.2322 | 6.7368 |
| Frozen fusion + prior GRU projection | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + prior GRU projection | 5 | 1.5987 | 3.5961 | 0.5663 | 0.2297 | 6.6904 |
| Frozen fusion + frozen GRU, anchor normalized | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + frozen GRU, anchor normalized | 5 | 1.5973 | 3.5937 | 0.5669 | 0.2295 | 6.6941 |
| Frozen fusion + updated GRU, anchor normalized | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + updated GRU, anchor normalized | 5 | 1.5952 | 3.5880 | 0.5681 | 0.2294 | 6.6886 |
| Frozen fusion + prior tree projection | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + prior tree projection | 5 | 1.5952 | 3.5994 | 0.5658 | 0.2287 | 6.7115 |

## Secondary conditional Q90 comparisons

These estimates condition on query DOC at or above each partition's source-training Q90.
They are tail diagnostics, not primary endpoints. The same paired station bootstrap is
applied to this tail population. Counts exclude repeated seed predictions; n < 20 in
any partition is marked unstable. A missing tail partition is not silently dropped.

| Comparison | Reference tail MAE | Candidate tail MAE | Difference [95% CI] | Reduction [95% CI] | Status; unstable |
|---|---:|---:|---:|---:|---|
| q90_context_tuned_vs_frozen_anchor_k5 | 6.7002 | 6.6945 | -0.0057 [-0.0198, 0.0084] | 0.09% [-0.13, 0.28] | estimated; False |
| q90_context_tuned_vs_tree_k5 | 6.7182 | 6.6945 | -0.0238 [-0.0696, 0.0193] | 0.35% [-0.30, 0.96] | estimated; False |
| q90_fusion_tuned_vs_frozen_anchor_k5 | 6.6941 | 6.6886 | -0.0055 [-0.0197, 0.0087] | 0.08% [-0.13, 0.28] | estimated; False |
| q90_fusion_tuned_vs_tree_k5 | 6.7115 | 6.6886 | -0.0229 [-0.0690, 0.0205] | 0.34% [-0.32, 0.95] | estimated; False |

## Source training and selected adaptation

| Split | Seed | Best / run epoch | Initial / best validation MAE | GRU / decay distance | Trainable parameters |
|---|---:|---:|---:|---:|---:|
| 142 | 42 | 7 / 12 | 1.9152 / 1.9121 | 0.33640 / 0.04570 | 25,472 |
| 142 | 43 | 0 / 5 | 1.9249 / 1.9249 | 0.00000 / 0.00000 | 25,472 |
| 142 | 44 | 16 / 21 | 1.9262 / 1.9218 | 0.62472 / 0.08312 | 25,472 |
| 143 | 42 | 4 / 9 | 1.4619 / 1.4592 | 0.15722 / 0.01707 | 25,472 |
| 143 | 43 | 0 / 5 | 1.4781 / 1.4781 | 0.00000 / 0.00000 | 25,472 |
| 143 | 44 | 0 / 5 | 1.4863 / 1.4863 | 0.00000 / 0.00000 | 25,472 |
| 144 | 42 | 30 / 30 | 1.8740 / 1.8653 | 1.25688 / 0.14599 | 25,472 |
| 144 | 43 | 0 / 5 | 1.8603 / 1.8603 | 0.00000 / 0.00000 | 25,472 |
| 144 | 44 | 7 / 12 | 1.8637 / 1.8596 | 0.28673 / 0.03196 | 25,472 |

Epoch 0 was selected in 4/9 runs.
Distances measure selected GRU/decay parameter change from initialization; recorded losses
are finite. `memory_training.csv` retains the training code's nonfinite-gradient policy
and anchor-scale floor counts. Source anchor counts describe batches during the selected
epoch, not a common checkpoint, and are absent for epoch 0; validation counts describe
the selected checkpoint. These are source diagnostics, not outer-test performance.

| Updated adapter | K | Finite shape enabled | Constant selected |
|---|---:|---:|---:|
| Environmental + updated GRU, anchor normalized | 3 | 9/9 | 0/9 |
| Environmental + updated GRU, anchor normalized | 5 | 9/9 | 0/9 |
| Frozen fusion + updated GRU, anchor normalized | 3 | 9/9 | 0/9 |
| Frozen fusion + updated GRU, anchor normalized | 5 | 9/9 | 0/9 |

## Reading the evidence

Updated versus frozen-anchor GRU is the direct recurrent-training comparison. Updated
versus prior GRU combines recurrent training with the normalization change; frozen-anchor
versus prior GRU reports that normalization change separately. Each complete adapter
selects its own alpha and ridge on source validation, so these are not fixed-alpha or
fixed-ridge ablations. Finite ridge enables a shape correction but does not guarantee a
nonzero correction at every station. The tree representation has the same two-dimensional
output and K support observations, but is not matched for trainable parameter count.

Only the existing GRUCell and observation-decay parameters are updated. K = 0 remains
the frozen base. Source environmental residuals use station-fold-fitted forests, while
spatial weights and normalization retain earlier source training. Source validation selects
the recurrent epoch and final adapter parameters; this is conditional development evaluation.

Each recurrent window respects temporal order, but calendar-anchor normalization can use
later covariates and support observations can follow query dates. The overall product is
retrospective reconstruction. No attention, river-message or statistical-causal claim follows
from this comparison. Seeds and repeated stations are not additional ecological samples.

All K curves, run/partition metrics, paired directions, station responses, adapter selections
and tail counts are saved with the report. Historical results remain available unchanged.
