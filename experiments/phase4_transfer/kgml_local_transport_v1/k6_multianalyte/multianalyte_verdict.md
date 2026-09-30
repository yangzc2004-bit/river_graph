# K6 cross-analyte KGML verdict

Positive gain means the upstream message residual improves on the local temporal residual.

         analyte              mask  n_cells  n_stations  local_mae  message_mae  gain_mae    ci_low   ci_high  gain_pct
              ph        e2a_strict     2081          67   0.193448     0.199385 -0.005937 -0.009021 -0.002572 -3.069066
              ph e3_spatial_seed42     2521          43   0.269833     0.268927  0.000906 -0.000431  0.001907  0.335685
spec_conductance        e2a_strict     2111          67 102.195883   102.772109 -0.576226 -2.118699  1.094265 -0.563845
spec_conductance e3_spatial_seed42     2511          43 396.405911   397.832380 -1.426469 -3.549205  1.411904 -0.359851

The comparison is paired by station-month and averaged over three training seeds.
