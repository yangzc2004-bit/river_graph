# ExtraTrees context + residual hybrid

Context model selection is validation-only and performed separately per seed and family. For temporal families, the transformed-space blend weight is selected on validation predictions; test predictions are then evaluated once. Missing runs are listed in `missing_runs.csv`.

## Family means

             mask  n_seeds  context_mae  residual_mae  hybrid_mae  alpha_mean  alpha_sd  hybrid_gain_vs_context_pct
    e1_r20_seed42        5     1.436170           NaN    1.436170        0.00  0.000000                    0.000000
       e2a_strict        5     1.047455      0.945607    0.934451        0.69  0.071063                   10.788504
      e2b_partial        5     1.016471      0.794807    0.816854        0.69  0.071063                   19.638276
e3_spatial_seed42        5     2.545262           NaN    2.545262        0.00  0.000000                    0.000000

## Pooled seed-family view

 n_seed_family_rows  pooled_context_mae  pooled_hybrid_mae  pooled_hybrid_gain_vs_context_pct  cell_weighted_context_mae  cell_weighted_hybrid_mae  cell_weighted_hybrid_gain_vs_context_pct  mean_temporal_alpha
                 20            1.511339           1.433184                           5.171276                   1.544514                  1.489641                                   3.55278                 0.69
