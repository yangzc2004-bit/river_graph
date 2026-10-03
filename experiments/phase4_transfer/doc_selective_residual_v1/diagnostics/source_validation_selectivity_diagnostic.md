# Source-validation selectivity: native DOC encoder residual

## Scope and definitions

This diagnostic uses the existing 30- and 60-epoch encoder products only at the
fixed source-validation query cells. The five scheduled support months remain
reserved at K=0. Parquet reads have an explicit `cell in validation_query` filter;
the selected rows are checked against the query vector and the `val` role.
Dataset labels are indexed only at source-training cells and validation queries.
No target-test predictions, labels or endpoint analyses were read; no model was
fitted. Dataset/mask hashes and saved validation MAEs are checked.

Q90 is the frozen source-training quantile, with a tail defined by truth >= Q90.
Signed bias is prediction minus truth. False-Q90 rate is false high predictions
divided by the number of ordinary query cells (truth < Q90). Ordinary positive
error is mean(max(prediction - truth, 0)) over **all** ordinary cells, not only
overpredictions. Ordinary MAE equals positive plus negative error mass.

Scores pool query cells within each run, average three seeds within partition,
then give each partition equal weight. Station-equal MAE additionally averages
station MAEs within each run. Seeds repeat the same ecological observations;
counts in the run CSV are not summed across seeds. Integrated models use the
already selected K0 ecological mixture, without fitting new mixture weights.

## Current 60-epoch panel

| model | mae | tail_mae | ordinary_mae | tail_bias | ordinary_bias | false_q90_rate | station_equal_mae |
|---|---|---|---|---|---|---|---|
| context | 1.990984 | 9.984876 | 1.290994 | -9.462878 | 0.362010 | 0.009296 | 2.481343 |
| frozen | 1.886054 | 9.682740 | 1.215544 | -8.951613 | 0.170076 | 0.011994 | 2.364210 |
| frozen_integrated | 1.880345 | 9.748760 | 1.204494 | -9.048074 | 0.097241 | 0.012166 | 2.356433 |
| last_self | 1.883320 | 9.676844 | 1.213817 | -8.936109 | 0.171404 | 0.012970 | 2.363083 |
| last_self_ecology | 1.869505 | 9.634572 | 1.204062 | -8.841819 | 0.146393 | 0.014050 | 2.356640 |
| last_self_ecology_integrated | 1.865599 | 9.681652 | 1.196517 | -8.912956 | 0.098110 | 0.014222 | 2.351543 |
| last_self_integrated | 1.878048 | 9.738503 | 1.203565 | -9.027773 | 0.106722 | 0.013142 | 2.356448 |

All errors and biases are in mg/L; false-Q90 rates are proportions.

Relative to context, self-plus-ecology lowers ordinary MAE from 1.290994 to
1.204062, ordinary bias from +0.362010 to +0.146393, and ordinary positive-error
mass from 0.826502 to 0.675227. Tail MAE falls from 9.984876 to 9.634572 and tail
bias improves from -9.462878 to -8.841819. False-Q90 rate nonetheless rises from
0.009296 to 0.014050. The model therefore reduces ordinary overprediction in
aggregate while making more errors around the high-value decision threshold.
Those are different properties and should not be conflated.

## Earlier 30-epoch panel

| model | mae | tail_mae | ordinary_mae | ordinary_bias | false_q90_rate |
|---|---|---|---|---|---|
| context | 1.990984 | 9.984876 | 1.290994 | 0.362010 | 0.009296 |
| frozen | 1.903160 | 9.706278 | 1.231846 | 0.191549 | 0.011539 |
| frozen_integrated | 1.897620 | 9.771348 | 1.221032 | 0.120360 | 0.011710 |
| last_self | 1.897143 | 9.687264 | 1.228128 | 0.194984 | 0.011929 |
| last_self_ecology | 1.893030 | 9.670770 | 1.226815 | 0.192333 | 0.013186 |
| last_self_ecology_integrated | 1.888433 | 9.731514 | 1.217376 | 0.133218 | 0.013272 |
| last_self_integrated | 1.891866 | 9.749021 | 1.217865 | 0.130423 | 0.012101 |

Longer training improves both ordinary and tail MAE in the equal-partition
summary. For the ecology arm, ordinary positive-error mass falls from 0.709574
to 0.675227 and ordinary bias falls from +0.192333 to +0.146393, while false-Q90
rate rises modestly from 0.013186 to 0.014050. Thus the duration benefit is not a
uniform upward shift of all predictions.

## Partition differences

- **Partition 142:** extra ecology tuning versus frozen improves tail MAE
  (6.054286 to 5.970752), but slightly worsens ordinary MAE (1.214397 to
  1.219168), ordinary bias (+0.087630 to +0.126039), and false-Q90 rate
  (0.016003 to 0.020882). This is the clearest local sensitivity–specificity
  tradeoff.
- **Partition 143:** ecology tuning improves both ordinary and tail MAE over
  frozen. Ordinary positive-error mass falls (0.986841 to 0.918200), although
  false-Q90 rate rises slightly (0.017795 to 0.019084). Ecological integration
  further reduces ordinary error while sacrificing some tail accuracy.
- **Partition 144:** all residual arms improve ordinary MAE over context, and
  ordinary signed bias is already negative. Tail MAE remains about 15.65 mg/L.
  Adding a universal ordinary-overprediction penalty would not directly address
  this partition's dominant high-value underprediction.

Full per-partition metrics and counts are saved separately. These validation
labels already selected checkpoints and mixture strengths, so the decomposition
supports the next development experiment rather than a new confirmatory claim.

## Source-station sampling concentration

| partition | n_source_stations | n_source_cells | station_count_median | station_count_max | top_decile_cell_share | effective_source_stations_cell_loss | effective_source_stations_tail_weight_two |
|---|---|---|---|---|---|---|---|
| 142 | 232 | 15958 | 37.000000 | 409 | 0.397857 | 105.386819 | 115.298093 |
| 143 | 232 | 15241 | 36.500000 | 320 | 0.391969 | 109.039312 | 118.706934 |
| 144 | 232 | 13727 | 36.000000 | 409 | 0.381438 | 110.100640 | 120.647762 |

The top decile rounds upward to 24 of 232 stations. Those stations contribute
38–40% of training cells; median station counts are only 36–37. Cell-weighted
loss therefore gives substantially greater influence to densely sampled source
stations. The effective station count here is the inverse squared sum of each
station's objective-weight share, not an inferential sample size.

Tail weight two increases the effective station count modestly, rather than
making station concentration worse. It should not be blamed for this sampling
imbalance without a matched loss comparison. Station-equal validation MAE is
also appreciably larger than pooled MAE, which shows that station-level errors
are distributed unevenly; it does not alone establish that reweighting will help.

## Recommended next experiment

**Test source-station-balanced training against the current cell-weighted loss,
keeping tail weight two, encoder mode, 60-epoch cap, patience and learning rates
fixed.** Continue choosing checkpoint and scale by the same pooled validation
MAE, and retain station-equal/tail/ordinary diagnostics as explanatory outputs.
A direct implementation gives each source station equal total loss weight,
with the existing tail weights normalized within station; its objective is the
mean of station-wise tail-weighted MAEs. Use a fixed whole-source normalization
so minibatch composition does not silently change the objective.

This tests a concrete mismatch between uneven training frequency and prediction
at new stations. It is not a claim that balancing must win under the existing
pooled endpoint.

The alternatives are less directly supported by these source-validation data:

1. **Extra ordinary-overprediction pressure:** useful as a focused follow-up
   if a remaining false-high cost matters, but broad overprediction already
   decreases and partition 144 ordinary predictions are biased downward.
2. **Tail weight one:** a clean future ablation of sensitivity against false
   positives, but current tail bias remains strongly negative and source
   station diversity slightly increases under weight two. Removing the tail
   emphasis is not the clearest first response to the observed imbalance.

No new model or training is launched by this diagnostic.
