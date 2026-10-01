# ExtraTrees context + residual hybrid

Context model selection is validation-only and performed separately per seed and family. For temporal families, the transformed-space blend weight is selected on validation predictions; test predictions are then evaluated once. Missing runs are listed in `missing_runs.csv`.

## Family means

             mask  n_seeds  context_mae  residual_mae  hybrid_mae  alpha_mean  alpha_sd  hybrid_gain_vs_context_pct
    e1_r20_seed42        3     1.436003           NaN    1.436003        0.00  0.000000                    0.000000
       e2a_strict        3     1.041543      0.945470    0.933550        0.64  0.026458                   10.368554
      e2b_partial        1     1.010303      0.798263    0.824425        0.65       NaN                   18.398231
e3_spatial_seed42        3     2.532928           NaN    2.532928        0.00  0.000000                    0.000000

## Pooled seed-family view

 n_seed_family_rows  pooled_context_mae  pooled_hybrid_mae  pooled_hybrid_gain_vs_context_pct  cell_weighted_context_mae  cell_weighted_hybrid_mae  cell_weighted_hybrid_gain_vs_context_pct  mean_temporal_alpha
                 10            1.604173           1.553187                           3.178316                   1.603044                  1.567526                                  2.215665               0.6425
