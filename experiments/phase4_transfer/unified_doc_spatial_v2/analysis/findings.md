# DOC spatial adaptation: robust fusion and temporal-shape comparison

All nine saved expert packages are evaluated on the original fixed station queries.
The three partition outcomes were seen during development. This analysis evaluates the
saved source-selected predictions; it does not choose configurations from outer-test errors.

## Robust fusion

Negative paired MAE difference and positive relative reduction favor the named candidate.
Intervals use 5,000 joint whole-station bootstrap resamples. Seed losses are averaged
within each partition, followed by equal partition weighting.

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| fusion_vs_context_k0 | 1.9028 | 1.9136 | 0.0108 [-0.0208, 0.0410] | -0.57% [-2.16, 1.09] | 1/3; 1/9 |
| fusion_vs_v1_hybrid_k0 | 2.0015 | 1.9136 | -0.0879 [-0.1574, -0.0313] | 4.39% [1.62, 7.32] | 2/3; 6/9 |
| fusion_vs_context_k5 | 1.6094 | 1.6085 | -0.0009 [-0.0022, 0.0004] | 0.06% [-0.03, 0.13] | 1/3; 1/9 |
| fusion_vs_v1_hybrid_k5 | 1.6261 | 1.6085 | -0.0176 [-0.0424, 0.0045] | 1.08% [-0.28, 2.49] | 2/3; 5/9 |

## Shape adaptation

| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|
| context_gru_shape_vs_constant_k3 | 1.6437 | 1.6392 | -0.0045 [-0.0116, 0.0015] | 0.27% [-0.09, 0.68] | 3/3; 5/9 |
| context_tree_shape_vs_constant_k3 | 1.6437 | 1.6327 | -0.0110 [-0.0289, 0.0039] | 0.67% [-0.24, 1.74] | 3/3; 8/9 |
| context_gru_vs_tree_shape_k3 | 1.6327 | 1.6392 | 0.0065 [-0.0089, 0.0243] | -0.40% [-1.50, 0.53] | 0/3; 2/9 |
| context_gru_shape_vs_constant_k5 | 1.6094 | 1.5999 | -0.0094 [-0.0215, 0.0002] | 0.59% [-0.01, 1.31] | 3/3; 7/9 |
| context_tree_shape_vs_constant_k5 | 1.6094 | 1.5972 | -0.0122 [-0.0267, 0.0003] | 0.76% [-0.02, 1.65] | 3/3; 8/9 |
| context_gru_vs_tree_shape_k5 | 1.5972 | 1.5999 | 0.0028 [-0.0054, 0.0115] | -0.17% [-0.73, 0.33] | 0/3; 4/9 |
| fusion_gru_shape_vs_constant_k3 | 1.6392 | 1.6345 | -0.0047 [-0.0127, 0.0015] | 0.29% [-0.09, 0.75] | 3/3; 5/9 |
| fusion_tree_shape_vs_constant_k3 | 1.6392 | 1.6291 | -0.0101 [-0.0266, 0.0037] | 0.61% [-0.22, 1.61] | 3/3; 7/9 |
| fusion_gru_vs_tree_shape_k3 | 1.6291 | 1.6345 | 0.0053 [-0.0098, 0.0225] | -0.33% [-1.38, 0.59] | 1/3; 3/9 |
| fusion_gru_shape_vs_constant_k5 | 1.6085 | 1.5990 | -0.0095 [-0.0216, 0.0002] | 0.59% [-0.02, 1.32] | 3/3; 7/9 |
| fusion_tree_shape_vs_constant_k5 | 1.6085 | 1.5961 | -0.0124 [-0.0268, 0.0002] | 0.77% [-0.01, 1.66] | 3/3; 8/9 |
| fusion_gru_vs_tree_shape_k5 | 1.5961 | 1.5990 | 0.0029 [-0.0053, 0.0117] | -0.18% [-0.73, 0.32] | 0/3; 3/9 |

## Error profiles

| Predictor | K | MAE | RMSE | R² | Log MAE | Training-Q90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Environmental + constant correction | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + constant correction | 5 | 1.6094 | 3.6146 | 0.5621 | 0.2323 | 6.7436 |
| Environmental + GRU shape | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + GRU shape | 5 | 1.5999 | 3.6029 | 0.5650 | 0.2302 | 6.7127 |
| Environmental + tree shape | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Environmental + tree shape | 5 | 1.5972 | 3.6034 | 0.5650 | 0.2292 | 6.7190 |
| Robust fusion + constant correction | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Robust fusion + constant correction | 5 | 1.6085 | 3.6113 | 0.5627 | 0.2322 | 6.7368 |
| Robust fusion + GRU shape | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Robust fusion + GRU shape | 5 | 1.5990 | 3.5996 | 0.5657 | 0.2301 | 6.7060 |
| Robust fusion + tree shape | 0 | 1.9136 | 4.0633 | 0.4513 | 0.2853 | 7.7785 |
| Robust fusion + tree shape | 5 | 1.5961 | 3.6001 | 0.5657 | 0.2291 | 6.7122 |
| Version-1 calibrated ExtraTrees | 0 | 1.9028 | 4.0080 | 0.4668 | 0.2859 | 7.5586 |
| Version-1 calibrated ExtraTrees | 5 | 1.6094 | 3.6146 | 0.5621 | 0.2323 | 6.7436 |
| Version-1 calibrated hybrid | 0 | 2.0015 | 4.1222 | 0.4328 | 0.2938 | 7.8227 |
| Version-1 calibrated hybrid | 5 | 1.6261 | 3.5932 | 0.5669 | 0.2350 | 6.6067 |

## Source-selected mechanisms

Fusion selections: context 5/9; ridge 4/9.

| Shape arm | K | Finite shape enabled | Constant shape selected |
|---|---:|---:|---:|
| Environmental + GRU shape | 3 | 6/9 | 3/9 |
| Environmental + GRU shape | 5 | 7/9 | 2/9 |
| Environmental + tree shape | 3 | 9/9 | 0/9 |
| Environmental + tree shape | 5 | 8/9 | 1/9 |
| Robust fusion + GRU shape | 3 | 6/9 | 3/9 |
| Robust fusion + GRU shape | 5 | 7/9 | 2/9 |
| Robust fusion + tree shape | 3 | 9/9 | 0/9 |
| Robust fusion + tree shape | 5 | 8/9 | 1/9 |

Finite regularization enables a shape term; it does not guarantee a nonzero correction
at every station. Infinite regularization selects the constant-offset case. Full source
choices are in `fusion_choices.csv` and `adapter_choices.csv`, including each alpha.

## Reading the comparisons

Fusion versus context measures the benefit of robust expert combination. Fusion versus
the version-1 hybrid measures the change from the earlier unrestricted selection procedure.
A shape-versus-constant comparison compares the complete temporal-shape adapter with the constant
adapter, allowing each its own source-selected level shrinkage; it is not a fixed-alpha
ablation of the shape term. GRU-versus-tree shape compares two equally sized
source-trained representations under the matched support adapter; it does not establish
a river-message or causal effect. A null result is retained alongside improvements.

K = 0 shape predictions equal their own unadapted base. K > 0 support labels may postdate
query months: this is retrospective reconstruction. Source-station meta-CV is conditional
on the previously selected frozen experts, rather than an independent evaluation of
their full fitting procedure.

The complete K curves are in `k_curves.csv`; per-run Q90 sample counts and unstable-tail
flags are in `run_metrics.csv`. Partition metrics retain unique query counts; seeds and
repeated stations are not additional ecological observations. `split_seed_directions.csv`
retains every result and `station_heterogeneity.csv` describes station responses.
