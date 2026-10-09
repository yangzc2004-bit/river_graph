# Unified DOC spatial reconstruction: PARTIAL

Completed runs: 3/9. Three planned spatial partitions (142–144), three training seeds (42–44), four support budgets (0, 1, 3, 5), and four paired arms.

## Main comparisons

MAEs average individual-seed losses within each partition and then weight partitions equally. Negative paired ΔMAE and positive relative reduction favor the candidate. Intervals resample global station IDs jointly across partitions; they preserve each partition's cell weighting and the repeated-station dependence.

| Comparison | Reference MAE | Candidate MAE | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions |
|---|---:|---:|---:|---:|---:|
| Hybrid vs ExtraTrees; Both calibrated, K = 5 | 1.5768 | 1.5647 | -0.0121 [-0.0240, 0.0003] | 0.77% [-0.02, 1.33] | 1/1 |
| Hybrid vs ExtraTrees; No target support, K = 0 | 1.9906 | 1.9201 | -0.0705 [-0.1240, -0.0226] | 3.54% [1.20, 5.88] | 1/1 |
| ExtraTrees calibration; K = 5 vs K = 0 | 1.9906 | 1.5768 | -0.4137 [-0.7623, -0.1821] | 20.78% [10.79, 28.89] | 1/1 |
| Hybrid calibration; K = 5 vs K = 0 | 1.9201 | 1.5647 | -0.3554 [-0.6661, -0.1528] | 18.51% [9.49, 26.24] | 1/1 |

## Interpretation

At K = 5, the paired interval spans zero; the evaluated partitions do not establish an additional hybrid improvement beyond calibrating ExtraTrees.
The hybrid equals its context prediction in 0/3 completed split–seed runs (numerical tolerance 1e-10 mg/L). Calibration gains in such runs are not neural gains. The component comparison does not isolate any one neural feature or river-message mechanism.
This is an interim analysis. Available seeds are averaged within each completed partition; missing runs can change all pooled estimates and intervals. Missing pairs: [{'split_seed': 143, 'seed': 42}, {'split_seed': 143, 'seed': 43}, {'split_seed': 143, 'seed': 44}, {'split_seed': 144, 'seed': 42}, {'split_seed': 144, 'seed': 43}, {'split_seed': 144, 'seed': 44}].

## K curves

| Model | K | MAE | Partition SD |
|---|---:|---:|---:|
| ExtraTrees | 0 | 1.9906 | NA |
| ExtraTrees + calibration | 0 | 1.9906 | NA |
| Hybrid | 0 | 1.9201 | NA |
| Hybrid + calibration | 0 | 1.9201 | NA |
| ExtraTrees | 1 | 1.9906 | NA |
| ExtraTrees + calibration | 1 | 1.8740 | NA |
| Hybrid | 1 | 1.9201 | NA |
| Hybrid + calibration | 1 | 1.8189 | NA |
| ExtraTrees | 3 | 1.9906 | NA |
| ExtraTrees + calibration | 3 | 1.5824 | NA |
| Hybrid | 3 | 1.9201 | NA |
| Hybrid + calibration | 3 | 1.6248 | NA |
| ExtraTrees | 5 | 1.9906 | NA |
| ExtraTrees + calibration | 5 | 1.5768 | NA |
| Hybrid | 5 | 1.9201 | NA |
| Hybrid + calibration | 5 | 1.5647 | NA |

## Supporting analyses

- `run_metrics.csv`: MAE, RMSE, R², log-space MAE and train-Q90 tail MAE. Tail counts are unique query cells per run; n < 20 is marked unstable.
- `split_seed_consistency.csv`: every paired split/seed outcome, including failures.
- `station_heterogeneity.csv`: within-partition seed-mean station effects. A station repeated in partitions is not an independent ecological replicate.
- `stratified_by_split.csv` and `stratified_summary.csv`: predictor-only descriptive novelty tertiles and presence/absence of visible upstream support. Empty strata are not assigned a synthetic zero; the summary records contributing partitions.
- `fallback_diagnostics.csv`: numerical context equivalence and saved adapter metadata.

The common support/query design evaluates retrospective reconstruction. These are new partitions of the existing cohort, not independent external basins. Station-bootstrap intervals are conditional on the selected partitions and fitted models.
