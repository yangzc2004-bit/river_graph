# Unified DOC spatial reconstruction: COMPLETE

Completed runs: 9/9. Three planned spatial partitions (142–144), three training seeds (42–44), four support budgets (0, 1, 3, 5), and four paired arms.

## Main comparisons

MAEs average individual-seed losses within each partition and then weight partitions equally. Negative paired ΔMAE and positive relative reduction favor the candidate. Intervals resample global station IDs jointly across partitions; they preserve each partition's cell weighting and the repeated-station dependence.

| Comparison | Reference MAE | Candidate MAE | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions |
|---|---:|---:|---:|---:|---:|
| Hybrid vs ExtraTrees; Both calibrated, K = 5 | 1.6094 | 1.6261 | 0.0167 [-0.0054, 0.0412] | -1.04% [-2.50, 0.33] | 1/3 |
| Hybrid vs ExtraTrees; No target support, K = 0 | 1.9028 | 2.0015 | 0.0987 [0.0333, 0.1722] | -5.19% [-8.71, -1.74] | 1/3 |
| ExtraTrees calibration; K = 5 vs K = 0 | 1.9028 | 1.6094 | -0.2934 [-0.4457, -0.1744] | 15.42% [9.83, 21.05] | 3/3 |
| Hybrid calibration; K = 5 vs K = 0 | 2.0015 | 1.6261 | -0.3755 [-0.5286, -0.2504] | 18.76% [13.78, 23.72] | 3/3 |

## Interpretation

At K = 5, the paired interval spans zero; the evaluated partitions do not establish an additional hybrid improvement beyond calibrating ExtraTrees.
The hybrid equals its context prediction in 0/9 completed split–seed runs (numerical tolerance 1e-10 mg/L). Calibration gains in such runs are not neural gains. The component comparison does not isolate any one neural feature or river-message mechanism.

## K curves

| Model | K | MAE | Partition SD |
|---|---:|---:|---:|
| ExtraTrees | 0 | 1.9028 | 0.1573 |
| ExtraTrees + calibration | 0 | 1.9028 | 0.1573 |
| Hybrid | 0 | 2.0015 | 0.1676 |
| Hybrid + calibration | 0 | 2.0015 | 0.1676 |
| ExtraTrees | 1 | 1.9028 | 0.1573 |
| ExtraTrees + calibration | 1 | 1.8666 | 0.1339 |
| Hybrid | 1 | 2.0015 | 0.1676 |
| Hybrid + calibration | 1 | 1.9431 | 0.2175 |
| ExtraTrees | 3 | 1.9028 | 0.1573 |
| ExtraTrees + calibration | 3 | 1.6437 | 0.0587 |
| Hybrid | 3 | 2.0015 | 0.1676 |
| Hybrid + calibration | 3 | 1.6747 | 0.0616 |
| ExtraTrees | 5 | 1.9028 | 0.1573 |
| ExtraTrees + calibration | 5 | 1.6094 | 0.0345 |
| Hybrid | 5 | 2.0015 | 0.1676 |
| Hybrid + calibration | 5 | 1.6261 | 0.0700 |

## Supporting analyses

- `run_metrics.csv`: MAE, RMSE, R², log-space MAE and train-Q90 tail MAE. Tail counts are unique query cells per run; n < 20 is marked unstable.
- `split_seed_consistency.csv`: every paired split/seed outcome, including failures.
- `station_heterogeneity.csv`: within-partition seed-mean station effects. A station repeated in partitions is not an independent ecological replicate.
- `stratified_by_split.csv` and `stratified_summary.csv`: predictor-only descriptive novelty tertiles and presence/absence of visible upstream support. Empty strata are not assigned a synthetic zero; the summary records contributing partitions.
- `fallback_diagnostics.csv`: numerical context equivalence and saved adapter metadata.

The common support/query design evaluates retrospective reconstruction. These are new partitions of the existing cohort, not independent external basins. Station-bootstrap intervals are conditional on the selected partitions and fitted models.
