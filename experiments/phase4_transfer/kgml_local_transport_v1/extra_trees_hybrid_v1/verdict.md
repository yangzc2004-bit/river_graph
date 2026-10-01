# ExtraTrees context + residual hybrid

Context model selection is validation-only and performed separately per seed and family. For temporal families, the transformed-space blend weight is selected on validation predictions; test predictions are then evaluated once. Missing runs are listed in `missing_runs.csv`.

## Family means

             mask  n_seeds  context_mae  residual_mae  hybrid_mae  alpha_mean  alpha_sd  hybrid_gain_vs_context_pct
    e1_r20_seed42        3     1.436003           NaN    1.436003        0.00  0.000000                    0.000000
       e2a_strict        3     1.041543      0.945470    0.933550        0.64  0.026458                   10.368554
      e2b_partial        3     1.012201      0.795166    0.823044        0.64  0.026458                   18.687689
e3_spatial_seed42        3     2.532928           NaN    2.532928        0.00  0.000000                    0.000000

## Pooled seed-family view

 n_seed_family_rows  pooled_context_mae  pooled_hybrid_mae  pooled_hybrid_gain_vs_context_pct  cell_weighted_context_mae  cell_weighted_hybrid_mae  cell_weighted_hybrid_gain_vs_context_pct  mean_temporal_alpha
                 12            1.505669           1.431381                           4.933853                   1.539743                  1.487562                                  3.388931                 0.64
