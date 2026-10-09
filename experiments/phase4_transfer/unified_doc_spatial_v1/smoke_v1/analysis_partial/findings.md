# Unified DOC spatial reconstruction: PARTIAL

**Implementation check only.** These runs do not use the planned 300-tree, 20-epoch confirmation budget. Their numbers test the reporting pipeline and are not confirmation evidence.

Completed runs: 1/9. Three planned spatial partitions (142–144), three training seeds (42–44), four support budgets (0, 1, 3, 5), and four paired arms.

## Main comparisons

MAEs average individual-seed losses within each partition and then weight partitions equally. Negative paired ΔMAE and positive relative reduction favor the candidate. Intervals resample global station IDs jointly across partitions; they preserve each partition's cell weighting and the repeated-station dependence.

| Comparison | Reference MAE | Candidate MAE | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions |
|---|---:|---:|---:|---:|---:|
| Hybrid vs ExtraTrees; Both calibrated, K = 5 | 1.6035 | 1.6012 | -0.0022 [-0.0153, 0.0128] | 0.14% [-0.81, 1.01] | 1/1 |
| Hybrid vs ExtraTrees; No target support, K = 0 | 2.0707 | 2.0418 | -0.0289 [-0.0819, 0.0272] | 1.40% [-1.34, 3.78] | 1/1 |
| ExtraTrees calibration; K = 5 vs K = 0 | 2.0707 | 1.6035 | -0.4673 [-0.8067, -0.2413] | 22.56% [13.41, 29.89] | 1/1 |
| Hybrid calibration; K = 5 vs K = 0 | 2.0418 | 1.6012 | -0.4406 [-0.7785, -0.2167] | 21.58% [12.93, 29.18] | 1/1 |

## Interpretation

The reduced-budget outputs were used only to check paired table construction, bootstrap execution, metadata handling and figure rendering. No scientific performance conclusion is assigned to this implementation check.
The hybrid equals its context prediction in 0/1 completed split–seed runs (numerical tolerance 1e-10 mg/L). Calibration gains in such runs are not neural gains. The component comparison does not isolate any one neural feature or river-message mechanism.
This is an interim analysis. Available seeds are averaged within each completed partition; missing runs can change all pooled estimates and intervals. Missing pairs: [{'split_seed': 142, 'seed': 43}, {'split_seed': 142, 'seed': 44}, {'split_seed': 143, 'seed': 42}, {'split_seed': 143, 'seed': 43}, {'split_seed': 143, 'seed': 44}, {'split_seed': 144, 'seed': 42}, {'split_seed': 144, 'seed': 43}, {'split_seed': 144, 'seed': 44}].

## K curves

| Model | K | MAE | Partition SD |
|---|---:|---:|---:|
| ExtraTrees | 0 | 2.0707 | NA |
| ExtraTrees + calibration | 0 | 2.0707 | NA |
| Hybrid | 0 | 2.0418 | NA |
| Hybrid + calibration | 0 | 2.0418 | NA |
| ExtraTrees | 1 | 2.0707 | NA |
| ExtraTrees + calibration | 1 | 1.9114 | NA |
| Hybrid | 1 | 2.0418 | NA |
| Hybrid + calibration | 1 | 1.9076 | NA |
| ExtraTrees | 3 | 2.0707 | NA |
| ExtraTrees + calibration | 3 | 1.6955 | NA |
| Hybrid | 3 | 2.0418 | NA |
| Hybrid + calibration | 3 | 1.6913 | NA |
| ExtraTrees | 5 | 2.0707 | NA |
| ExtraTrees + calibration | 5 | 1.6035 | NA |
| Hybrid | 5 | 2.0418 | NA |
| Hybrid + calibration | 5 | 1.6012 | NA |

## Supporting analyses

- `run_metrics.csv`: MAE, RMSE, R², log-space MAE and train-Q90 tail MAE. Tail counts are unique query cells per run; n < 20 is marked unstable.
- `split_seed_consistency.csv`: every paired split/seed outcome, including failures.
- `station_heterogeneity.csv`: within-partition seed-mean station effects. A station repeated in partitions is not an independent ecological replicate.
- `stratified_by_split.csv` and `stratified_summary.csv`: predictor-only descriptive novelty tertiles and presence/absence of visible upstream support. Empty strata are not assigned a synthetic zero; the summary records contributing partitions.
- `fallback_diagnostics.csv`: numerical context equivalence and saved adapter metadata.

The common support/query design evaluates retrospective reconstruction. These are new partitions of the existing cohort, not independent external basins. Station-bootstrap intervals are conditional on the selected partitions and fitted models.
