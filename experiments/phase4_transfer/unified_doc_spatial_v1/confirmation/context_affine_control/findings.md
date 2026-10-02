# Affine and local-forest component ablations

Additional controls; original four-arm primary results are unchanged.

Completed runs: 9/9. Source validation selects identity versus context-only log-affine recalibration and then independently selects the control's support shrinkage. A second control replaces the temporal residual expert R with its local forest B and fits C+B using the primary fusion/calibration class and validation populations.

| Comparison | Reference MAE | Candidate MAE | Reduction (%) [95% CI] |
|---|---:|---:|---:|
| hybrid_vs_affine_context_k0 | 1.9834 | 2.0015 | -0.91 [-2.23, 0.46] |
| hybrid_vs_affine_context_k1 | 1.9506 | 1.9431 | 0.38 [-1.24, 2.03] |
| hybrid_vs_affine_context_k3 | 1.6666 | 1.6747 | -0.48 [-1.43, 0.44] |
| hybrid_vs_affine_context_k5 | 1.6286 | 1.6261 | 0.16 [-0.81, 1.08] |
| affine_context_vs_original_et_k0 | 1.9028 | 1.9834 | -4.24 [-7.58, -0.87] |
| affine_context_vs_original_et_calibrated_k5 | 1.6094 | 1.6286 | -1.20 [-2.91, 0.36] |
| hybrid_vs_context_local_forest_k0 | 2.0097 | 2.0015 | 0.41 [0.12, 0.71] |
| hybrid_vs_context_local_forest_calibrated_k5 | 1.6232 | 1.6261 | -0.18 [-0.33, -0.03] |

Context-only affine recalibration was selected in 9/9 runs.

MAEs average individual-seed losses within partition, then weight partitions equally. Confidence intervals jointly resample station IDs across partitions and retain each partition's cell weighting. Positive reduction favors the named candidate.

At K > 0, both compared models use their own source-validation-selected station calibration. K = 0 compares unadapted predictions. The additional ablation separates the complete temporal expert's contribution from simple global recalibration; it does not isolate an individual neural mechanism.

The C+R versus C+B comparison at K = 0 and K = 5 isolates the addition of the trained temporal residual to the local forest, while allowing both combinations the same fusion candidates and station-calibration selection. The saved `local_forest_control` selection contains all C+B validation candidate scores. No additional forests or recurrent models were trained.

Support can postdate query months. The experiment concerns retrospective station reconstruction in the known cohort. Selection scores are training/selection diagnostics, not independent performance estimates.
