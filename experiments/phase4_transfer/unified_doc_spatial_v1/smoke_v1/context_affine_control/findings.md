# Affine and local-forest component ablations

Additional controls; original four-arm primary results are unchanged.

Completed runs: 1/9. Source validation selects identity versus context-only log-affine recalibration and then independently selects the control's support shrinkage. A second control replaces the temporal residual expert R with its local forest B and fits C+B using the primary fusion/calibration class and validation populations.

**Implementation check only.** Reduced-budget smoke outputs verify the analysis workflow. They do not establish scientific performance.

| Comparison | Reference MAE | Candidate MAE | Reduction (%) [95% CI] |
|---|---:|---:|---:|
| hybrid_vs_affine_context_k0 | 2.0285 | 2.0418 | -0.66 [-2.79, 1.29] |
| hybrid_vs_affine_context_k1 | 1.8836 | 1.9076 | -1.27 [-3.36, 0.82] |
| hybrid_vs_affine_context_k3 | 1.6823 | 1.6913 | -0.53 [-2.00, 0.82] |
| hybrid_vs_affine_context_k5 | 1.6016 | 1.6012 | 0.02 [-0.82, 0.89] |
| affine_context_vs_original_et_k0 | 2.0707 | 2.0285 | 2.04 [0.93, 2.99] |
| affine_context_vs_original_et_calibrated_k5 | 1.6035 | 1.6016 | 0.12 [-0.19, 0.39] |
| hybrid_vs_context_local_forest_k0 | 2.0421 | 2.0418 | 0.02 [-0.01, 0.04] |
| hybrid_vs_context_local_forest_calibrated_k5 | 1.6012 | 1.6012 | 0.00 [-0.00, 0.00] |

Context-only affine recalibration was selected in 1/1 runs.

MAEs average individual-seed losses within partition, then weight partitions equally. Confidence intervals jointly resample station IDs across partitions and retain each partition's cell weighting. Positive reduction favors the named candidate.

At K > 0, both compared models use their own source-validation-selected station calibration. K = 0 compares unadapted predictions. The additional ablation separates the complete temporal expert's contribution from simple global recalibration; it does not isolate an individual neural mechanism.

The C+R versus C+B comparison at K = 0 and K = 5 isolates the addition of the trained temporal residual to the local forest, while allowing both combinations the same fusion candidates and station-calibration selection. The saved `local_forest_control` selection contains all C+B validation candidate scores. No additional forests or recurrent models were trained.

Support can postdate query months. The experiment concerns retrospective station reconstruction in the known cohort. Selection scores are training/selection diagnostics, not independent performance estimates.
