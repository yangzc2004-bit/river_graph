# Matched 120-versus-60 epoch budget comparison

The duration extension was chosen from source-validation stopping trajectories before
examining the target results of either loss panel. The four loss definitions, patience,
initialization, learning rates, data, thresholds and fixed query populations are unchanged.
Each budget retains its own validation-selected checkpoint, residual scale, ecological
mixture and support adapter. Every arm and both positive and negative effects are shown.

All 36 shared training prefixes reproduce exactly. Prior runs whose patience
was already exhausted must reproduce their entire trajectory. Reference encoder and
ecological-affine products remain the same historical products under both budgets.

The reference root is `experiments/phase4_transfer/doc_selective_residual_v1`. Intervals use the same 5,000 joint station
bootstrap draws and equal-partition estimator as the main analysis. Repeated seeds
predict the same ecological samples. K0 constant and GRU-support predictions coincide,
so the duplicate constant K0 contrast is omitted. Recall gains must be read with false
alarms and ordinary/tail error rather than used to choose a preferred budget post hoc.

## Fixed paired duration contrasts

| Model / support path | K | Overall Delta MAE [95% CI] | Q90 Delta MAE [95% CI] | Ordinary Delta MAE [95% CI] | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|---:|
| tail2_constant | 5 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| tail2_gru_tuned_anchor | 0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| tail2_gru_tuned_anchor | 5 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| tail2_integrated_constant | 5 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| tail2_integrated_gru_tuned_anchor | 0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| tail2_integrated_gru_tuned_anchor | 5 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| mae_constant | 5 | +0.0004 [-0.0006, +0.0015] | +0.0014 [-0.0005, +0.0043] | +0.0002 [-0.0007, +0.0014] | -0.0316 [-0.1208, +0.0000] | +0.0031 [+0.0000, +0.0111] |
| mae_gru_tuned_anchor | 0 | -0.0000 [-0.0050, +0.0049] | +0.0048 [-0.0009, +0.0144] | -0.0005 [-0.0057, +0.0046] | -0.0316 [-0.1140, +0.0000] | -0.0031 [-0.0105, +0.0000] |
| mae_gru_tuned_anchor | 5 | +0.0003 [-0.0006, +0.0014] | +0.0017 [-0.0001, +0.0043] | +0.0002 [-0.0008, +0.0013] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| mae_integrated_constant | 5 | +0.0004 [-0.0006, +0.0015] | +0.0014 [-0.0005, +0.0043] | +0.0002 [-0.0007, +0.0014] | -0.0316 [-0.1208, +0.0000] | +0.0031 [+0.0000, +0.0111] |
| mae_integrated_gru_tuned_anchor | 0 | -0.0000 [-0.0050, +0.0049] | +0.0048 [-0.0009, +0.0144] | -0.0005 [-0.0057, +0.0046] | -0.0316 [-0.1140, +0.0000] | -0.0031 [-0.0105, +0.0000] |
| mae_integrated_gru_tuned_anchor | 5 | +0.0003 [-0.0006, +0.0014] | +0.0017 [-0.0001, +0.0043] | +0.0002 [-0.0008, +0.0013] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| selective_constant | 5 | -0.0001 [-0.0006, +0.0004] | +0.0004 [-0.0014, +0.0020] | -0.0001 [-0.0006, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| selective_gru_tuned_anchor | 0 | +0.0003 [-0.0007, +0.0014] | +0.0003 [-0.0020, +0.0023] | +0.0003 [-0.0008, +0.0015] | +0.0631 [+0.0000, +0.1792] | +0.0000 [+0.0000, +0.0000] |
| selective_gru_tuned_anchor | 5 | -0.0001 [-0.0006, +0.0004] | +0.0004 [-0.0014, +0.0021] | -0.0001 [-0.0006, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| selective_integrated_constant | 5 | +0.0049 [-0.0015, +0.0136] | +0.0357 [-0.0040, +0.0825] | +0.0019 [-0.0034, +0.0072] | +0.0000 [-0.1862, +0.1583] | -0.0092 [-0.0492, +0.0143] |
| selective_integrated_gru_tuned_anchor | 0 | +0.0003 [-0.0007, +0.0014] | +0.0003 [-0.0020, +0.0023] | +0.0003 [-0.0008, +0.0015] | +0.0631 [+0.0000, +0.1792] | +0.0000 [+0.0000, +0.0000] |
| selective_integrated_gru_tuned_anchor | 5 | -0.0014 [-0.0031, +0.0001] | -0.0040 [-0.0164, +0.0077] | -0.0011 [-0.0021, -0.0001] | -0.0316 [-0.1208, +0.0000] | -0.0061 [-0.0177, +0.0000] |
| station_constant | 5 | -0.0006 [-0.0013, +0.0001] | +0.0001 [-0.0035, +0.0027] | -0.0006 [-0.0015, +0.0001] | +0.0947 [+0.0000, +0.2755] | +0.0000 [-0.0090, +0.0092] |
| station_gru_tuned_anchor | 0 | -0.0005 [-0.0028, +0.0019] | +0.0002 [-0.0045, +0.0038] | -0.0006 [-0.0029, +0.0020] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| station_gru_tuned_anchor | 5 | +0.0042 [-0.0018, +0.0123] | +0.0306 [-0.0108, +0.0794] | +0.0017 [-0.0034, +0.0062] | +0.0631 [-0.1440, +0.2894] | -0.0031 [-0.0403, +0.0294] |
| station_integrated_constant | 5 | +0.0031 [-0.0034, +0.0103] | +0.0213 [-0.0127, +0.0585] | +0.0013 [-0.0054, +0.0069] | +0.0316 [-0.1747, +0.2384] | +0.0092 [-0.0169, +0.0438] |
| station_integrated_gru_tuned_anchor | 0 | -0.0005 [-0.0028, +0.0019] | +0.0002 [-0.0045, +0.0038] | -0.0006 [-0.0029, +0.0020] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| station_integrated_gru_tuned_anchor | 5 | -0.0003 [-0.0009, +0.0001] | +0.0000 [-0.0020, +0.0013] | -0.0004 [-0.0010, +0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

Negative error/FPR deltas favor the extended budget; positive recall deltas mean
more detected high observations. Raw/log error and bias comparisons, seed/partition
directions, and station gain/harm concentrations are retained in the CSVs.

## Source-validation duration and selection

| Arm | Reference total epochs | Extended total epochs | Reference best epoch range | Extended best epoch range | Fits selecting beyond reference cap | Mean validation MAE change |
|---|---:|---:|---:|---:|---:|---:|
| tail2 | 307 | 307 | 5–49 | 5–49 | 0/9 | +0.000000 |
| mae | 342 | 359 | 2–60 | 2–67 | 1/9 | -0.001735 |
| selective | 407 | 413 | 19–57 | 19–61 | 1/9 | -0.000273 |
| station | 304 | 323 | 0–60 | 0–69 | 1/9 | -0.000896 |

A higher cap does not force a longer fit or a later selected checkpoint. Improvements
in integrated products can include changed validation mixture/adapter choices; they
cannot be attributed entirely to additional neural optimization. This is a model
development comparison on the same previously examined station partitions.
