# K2 message gain by observation and hydrologic context

Positive gain means the message-only model has lower absolute error than the zero-message null.
Bins are descriptive diagnostics on the frozen test products; they were not used to select a model.

             mask               variable            level  n_cells  n_stations  mean_gain_mae    ci_low  ci_high  message_mae  null_mae  mean_abs_message_delta
       e2a_strict              age_group   never_observed       50           2       0.020060  0.000000 0.028657     0.559216  0.579276                0.015990
       e2a_strict              age_group         older_12     1752          60       0.043154  0.027898 0.059664     1.178336  1.221490                0.013547
       e2a_strict              age_group       recent_1_3       85          36       0.052356  0.025558 0.080541     0.955250  1.007606                0.017535
       e2a_strict              age_group    seasonal_4_12      336          57       0.053111  0.033584 0.074027     1.228192  1.281303                0.014809
       e2a_strict upstream_support_group             none     2223          69       0.044491  0.029112 0.061336     1.163416  1.207908                0.013945
       e2a_strict               flow_bin              0-1      505          25       0.040772  0.015537 0.067785     0.693258  0.734029                0.014187
       e2a_strict               flow_bin             1-10      150          14       0.024571  0.000000 0.073720     1.178635  1.203206                0.003633
       e2a_strict               flow_bin           10-100      331          19       0.018375  0.000025 0.050218     1.899286  1.917662                0.004497
       e2a_strict               flow_bin           100-1k      388          26       0.028219 -0.002866 0.064589     1.596618  1.624837                0.008046
       e2a_strict               flow_bin         10k-100k      335          18       0.071543  0.049991 0.090826     0.887048  0.958591                0.023616
       e2a_strict               flow_bin           1k-10k      377          25       0.059447  0.032915 0.087425     1.102771  1.162218                0.019537
       e2a_strict               flow_bin            >100k      137           6       0.081895  0.020793 0.115809     0.717712  0.799606                0.024844
e3_spatial_seed42              age_group   never_observed     2531          43       0.017047  0.003201 0.032255     2.957818  2.974866                0.005188
e3_spatial_seed42 upstream_support_group             none     1510          42       0.007471  0.000411 0.017138     3.466018  3.473489                0.002780
e3_spatial_seed42 upstream_support_group visible_upstream     1021          15       0.031210  0.002781 0.055822     2.206221  2.237431                0.008748
e3_spatial_seed42               flow_bin              0-1      871          27       0.005776 -0.001904 0.017664     3.689511  3.695287                0.006057
e3_spatial_seed42               flow_bin             1-10      257          22      -0.002376 -0.009240 0.003168     4.033614  4.031237                0.002034
e3_spatial_seed42               flow_bin           10-100      512          28       0.016222 -0.002204 0.034358     2.645031  2.661253                0.003154
e3_spatial_seed42               flow_bin           100-1k      387          24       0.009982 -0.001933 0.024455     2.329405  2.339386                0.003590
e3_spatial_seed42               flow_bin         10k-100k       32           3       0.057010 -0.046474 0.071953     1.931763  1.988773                0.010731
e3_spatial_seed42               flow_bin           1k-10k      188          13       0.024159  0.000005 0.040071     1.656703  1.680862                0.006189
e3_spatial_seed42               flow_bin            >100k      284           1       0.071098       NaN      NaN     2.137408  2.208506                0.009933

## Reading

The temporal mask has no visible-upstream support in the test query, so its upstream-support contrast is not identifiable.
The spatial mask provides the relevant upstream-support contrast. Intervals are station-clustered and should be read as diagnostic evidence, not a new primary endpoint.
