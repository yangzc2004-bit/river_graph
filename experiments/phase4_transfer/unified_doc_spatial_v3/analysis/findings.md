# DOC station adaptation: episodic representations and matched PCA

The complete nine-run panel compares frozen environmental/fusion predictions with
source-trained, two-dimensional episodic representations and their matched PCA controls.
Partitions 142–144 have been seen during development; these are development results
on same-cohort station holdouts, not independent external-basin confirmation.

Negative paired MAE difference and positive relative reduction favor the candidate.
Intervals use 5,000 joint whole-station bootstrap resamples, preserving repeated
station identities across partitions. Seed losses are averaged within partitions,
followed by equal partition weighting. The analysis makes no test-based selections.

## Episodic training versus matched PCA

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_gru_episodic_vs_pca_k3 | 1.6392 | 1.6395 | 0.0003 [-0.0126, 0.0128] | -0.02% [-0.80, 0.75] | 1/3; 5/9 |
| context_gru_episodic_vs_pca_k5 | 1.5999 | 1.5995 | -0.0004 [-0.0074, 0.0065] | 0.03% [-0.40, 0.45] | 2/3; 4/9 |
| context_tree_episodic_vs_pca_k3 | 1.6327 | 1.6337 | 0.0010 [-0.0034, 0.0047] | -0.06% [-0.29, 0.20] | 1/3; 3/9 |
| context_tree_episodic_vs_pca_k5 | 1.5972 | 1.5962 | -0.0009 [-0.0030, 0.0012] | 0.06% [-0.07, 0.18] | 2/3; 4/9 |
| fusion_gru_episodic_vs_pca_k3 | 1.6345 | 1.6358 | 0.0013 [-0.0110, 0.0133] | -0.08% [-0.84, 0.65] | 1/3; 5/9 |
| fusion_gru_episodic_vs_pca_k5 | 1.5990 | 1.5987 | -0.0003 [-0.0074, 0.0065] | 0.02% [-0.41, 0.44] | 2/3; 4/9 |
| fusion_tree_episodic_vs_pca_k3 | 1.6291 | 1.6307 | 0.0016 [-0.0032, 0.0057] | -0.10% [-0.36, 0.19] | 1/3; 3/9 |
| fusion_tree_episodic_vs_pca_k5 | 1.5961 | 1.5952 | -0.0009 [-0.0030, 0.0012] | 0.06% [-0.07, 0.19] | 2/3; 4/9 |

## GRU versus tree representations

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_gru_vs_tree_episodic_k3 | 1.6337 | 1.6395 | 0.0058 [-0.0148, 0.0272] | -0.36% [-1.73, 0.86] | 1/3; 3/9 |
| context_gru_vs_tree_episodic_k5 | 1.5962 | 1.5995 | 0.0033 [-0.0077, 0.0142] | -0.21% [-0.90, 0.47] | 1/3; 2/9 |
| fusion_gru_vs_tree_episodic_k3 | 1.6307 | 1.6358 | 0.0050 [-0.0148, 0.0252] | -0.31% [-1.59, 0.87] | 1/3; 3/9 |
| fusion_gru_vs_tree_episodic_k5 | 1.5952 | 1.5987 | 0.0035 [-0.0075, 0.0143] | -0.22% [-0.90, 0.45] | 1/3; 2/9 |

## Complete episodic adapters versus constant correction

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_gru_episodic_vs_constant_k5 | 1.6094 | 1.5995 | -0.0099 [-0.0252, 0.0033] | 0.61% [-0.21, 1.52] | 2/3; 8/9 |
| context_tree_episodic_vs_constant_k5 | 1.6094 | 1.5962 | -0.0131 [-0.0283, 0.0003] | 0.82% [-0.02, 1.76] | 3/3; 8/9 |
| fusion_gru_episodic_vs_constant_k5 | 1.6085 | 1.5987 | -0.0098 [-0.0252, 0.0033] | 0.61% [-0.21, 1.52] | 2/3; 8/9 |
| fusion_tree_episodic_vs_constant_k5 | 1.6085 | 1.5952 | -0.0133 [-0.0284, 0.0001] | 0.83% [-0.01, 1.77] | 3/3; 8/9 |

## Error profiles

| Predictor | K | MAE | RMSE | R² | Log MAE | Training-Q90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Environmental + constant correction | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + constant correction | 5 | 1.6094 | 3.6146 | 0.5621 | 0.2323 | 6.7436 |
| Environmental + episodic GRU | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + episodic GRU | 5 | 1.5995 | 3.5993 | 0.5657 | 0.2297 | 6.6969 |
| Environmental + GRU PCA | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + GRU PCA | 5 | 1.5999 | 3.6029 | 0.5650 | 0.2302 | 6.7127 |
| Environmental + episodic tree | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + episodic tree | 5 | 1.5962 | 3.6027 | 0.5652 | 0.2288 | 6.7182 |
| Environmental + tree PCA | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + tree PCA | 5 | 1.5972 | 3.6034 | 0.5650 | 0.2292 | 6.7190 |
| Frozen fusion + constant correction | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + constant correction | 5 | 1.6085 | 3.6113 | 0.5627 | 0.2322 | 6.7368 |
| Frozen fusion + episodic GRU | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + episodic GRU | 5 | 1.5987 | 3.5961 | 0.5663 | 0.2297 | 6.6904 |
| Frozen fusion + GRU PCA | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + GRU PCA | 5 | 1.5990 | 3.5996 | 0.5657 | 0.2301 | 6.7060 |
| Frozen fusion + episodic tree | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + episodic tree | 5 | 1.5952 | 3.5994 | 0.5658 | 0.2287 | 6.7115 |
| Frozen fusion + tree PCA | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Frozen fusion + tree PCA | 5 | 1.5961 | 3.6001 | 0.5657 | 0.2291 | 6.7122 |

## Source-selected training and adaptation

Projector initialization, best validation epoch/loss and displacement from PCA are
recorded for each GRU/tree run in `projector_training.csv`. The table below averages
the nine source-validation training summaries; these are not test-error estimates.

| Representation | Mean best epoch | Initial validation MAE | Best validation MAE | Mean subspace displacement |
|---|---:|---:|---:|---:|
| GRU | 24.00 | 1.7855 | 1.7506 | 1.1391 |
| TREE | 11.22 | 1.7643 | 1.7580 | 0.6727 |

Subspace displacement is the Frobenius distance between the final and initial
projection matrices WᵀW, making it invariant to rotations within the two output dimensions.

| Adapter | K | Finite shape enabled | Constant shape selected |
|---|---:|---:|---:|
| Environmental + episodic GRU | 3 | 9/9 | 0/9 |
| Environmental + episodic GRU | 5 | 9/9 | 0/9 |
| Environmental + GRU PCA | 3 | 6/9 | 3/9 |
| Environmental + GRU PCA | 5 | 7/9 | 2/9 |
| Environmental + episodic tree | 3 | 9/9 | 0/9 |
| Environmental + episodic tree | 5 | 8/9 | 1/9 |
| Environmental + tree PCA | 3 | 9/9 | 0/9 |
| Environmental + tree PCA | 5 | 8/9 | 1/9 |
| Frozen fusion + episodic GRU | 3 | 9/9 | 0/9 |
| Frozen fusion + episodic GRU | 5 | 9/9 | 0/9 |
| Frozen fusion + GRU PCA | 3 | 6/9 | 3/9 |
| Frozen fusion + GRU PCA | 5 | 7/9 | 2/9 |
| Frozen fusion + episodic tree | 3 | 9/9 | 0/9 |
| Frozen fusion + episodic tree | 5 | 8/9 | 1/9 |
| Frozen fusion + tree PCA | 3 | 9/9 | 0/9 |
| Frozen fusion + tree PCA | 5 | 8/9 | 1/9 |

Finite regularization enables a shape term, but does not imply a nonzero correction
at every station. Infinite regularization selects a constant correction; all alpha and
ridge choices are retained in `adapter_choices.csv`.

## Interpretation

The primary comparison asks whether supervised episodic projection improves the complete
support adapter relative to its within-run PCA representation. Each representation keeps
its own source-selected level shrinkage and ridge strength, so this is not a fixed-alpha
ablation of the shape term. The GRU-versus-tree comparison uses equally sized projections
and the same support budget; it does not isolate river messages or establish causal effects.

The five source context forests are refitted without held-station labels. GRU/tree feature
extractors remain frozen from prior training and are evaluated under label-hidden source
views. Their weights are not refitted as out-of-fold experts. Source validation selects the
projection epoch and final adapter parameters; this is conditional development evaluation.

K = 0 is exactly the frozen base. Support observations can postdate query months: this is
retrospective reconstruction. Q90 thresholds come from each run's source training labels.
Unique query counts and unstable-tail flags are in `run_metrics.csv`; seeds and repeated
stations do not increase the ecological sample size. Complete K curves, all partition/seed
directions and station responses remain available alongside the aggregate comparisons.
