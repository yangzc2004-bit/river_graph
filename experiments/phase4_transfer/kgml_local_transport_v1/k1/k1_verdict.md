# K1 Local--Transport KGML verdict

Completed runs: 36

## Mean test MAE by arm

                  arm               mask      mean       std  count
0               h2x_t         e2a_strict  1.142195  0.029838      3
1               h2x_t  e3_spatial_seed42  3.490089  0.213001      3
2       residual_both         e2a_strict  0.985162  0.039127      3
3       residual_both  e3_spatial_seed42  2.957176  0.030223      3
4      residual_nomsg         e2a_strict  0.981643  0.036443      3
5      residual_nomsg  e3_spatial_seed42  2.958316  0.031134      3
6   residual_upstream         e2a_strict  0.982704  0.038968      3
7   residual_upstream  e3_spatial_seed42  2.957746  0.030643      3
8          rf_context         e2a_strict  1.088988  0.031927      3
9          rf_context  e3_spatial_seed42  2.683758  0.026928      3
10           rf_local         e2a_strict  1.207908  0.050070      3
11           rf_local  e3_spatial_seed42  2.974866  0.034610      3

## Paired station bootstrap

             mask                          comparison  mae_candidate  mae_reference  gain_mae   gain_pct    ci_low   ci_high
       e2a_strict                   h2x_t_vs_rf_local       1.110511       1.204895  0.096461   8.005780 -0.268381  0.416585
       e2a_strict       residual_upstream_vs_rf_local       0.977308       1.204895  0.263871  21.899946  0.151634  0.377178
       e2a_strict           residual_both_vs_rf_local       0.979751       1.204895  0.265342  22.022024  0.149429  0.382865
       e2a_strict          residual_nomsg_vs_rf_local       0.976260       1.204895  0.268851  22.313240  0.152347  0.386167
       e2a_strict                 h2x_t_vs_rf_context       1.110511       1.083995 -0.019198  -1.771063 -0.211905  0.158828
       e2a_strict     residual_upstream_vs_rf_context       0.977308       1.083995  0.148212  13.672739 -0.108654  0.440383
       e2a_strict         residual_both_vs_rf_context       0.979751       1.083995  0.149683  13.808433 -0.104598  0.438609
       e2a_strict        residual_nomsg_vs_rf_context       0.976260       1.083995  0.153192  14.132129 -0.099774  0.441090
       e2a_strict residual_upstream_vs_residual_nomsg       0.977308       0.976260 -0.004980  -0.510086 -0.010751  0.000611
       e2a_strict     residual_both_vs_residual_nomsg       0.979751       0.976260 -0.003509  -0.359417 -0.009608  0.001919
e3_spatial_seed42                   h2x_t_vs_rf_local       3.430782       2.971373 -0.753699 -25.365329 -1.301838 -0.252933
e3_spatial_seed42       residual_upstream_vs_rf_local       2.953712       2.971373  0.005265   0.177176 -0.010703  0.020394
e3_spatial_seed42           residual_both_vs_rf_local       2.953159       2.971373  0.005422   0.182476 -0.011431  0.021375
e3_spatial_seed42          residual_nomsg_vs_rf_local       2.954307       2.971373  0.004988   0.167855 -0.010855  0.019997
e3_spatial_seed42                 h2x_t_vs_rf_context       3.430782       2.679058 -1.052993 -39.304615 -1.607302 -0.582457
e3_spatial_seed42     residual_upstream_vs_rf_context       2.953712       2.679058 -0.294030 -10.975136 -0.497208 -0.091424
e3_spatial_seed42         residual_both_vs_rf_context       2.953159       2.679058 -0.293873 -10.969257 -0.497662 -0.090848
e3_spatial_seed42        residual_nomsg_vs_rf_context       2.954307       2.679058 -0.294307 -10.985474 -0.497522 -0.091629
e3_spatial_seed42 residual_upstream_vs_residual_nomsg       2.953712       2.954307  0.000277   0.009375  0.000093  0.000506
e3_spatial_seed42     residual_both_vs_residual_nomsg       2.953159       2.954307  0.000434   0.014706 -0.000592  0.001407

## Scientific reading

The residual arms improve over RF-local, especially in temporal extrapolation. The upstream, bidirectional, and no-message residual arms are nearly identical. Therefore K1 supports residual correction, but does not isolate an additional river-message contribution. RF-context remains the strongest spatial baseline. K2 should test a simpler source-isolation design or process-state representation before adding lag complexity.
