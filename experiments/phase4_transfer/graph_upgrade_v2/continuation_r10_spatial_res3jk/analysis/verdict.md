# DOC spatial res3-jk pilot

A 3-layer residual spatial trunk with jumping knowledge was compared with the existing observation-aware GRU trunk on the same DOC E3 spatial holdout and seeds 42--44. The comparison is exploratory and uses the same evaluation script.

   comparison  n_seeds  m1_mae_mean  res3jk_mae_mean  mae_gain_pct_mean  m1_rmse_mean  res3jk_rmse_mean  rmse_gain_pct_mean  m1_q90_mae_mean  res3jk_q90_mae_mean  q90_gain_pct_mean  seed_mae_better
res3_jk_vs_m1        3     3.490092         3.456339            0.94659      6.827275          6.823613            0.043418              NaN            11.256922                NaN                2

The mean gain is modest and one seed is worse; this is a candidate spatial improvement, not evidence that deeper message passing solves the spatial gap.
