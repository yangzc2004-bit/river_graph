# K2 message gain by observation and hydrologic context

Positive gain means the message-only model has lower absolute error than the zero-message null.
Bins are descriptive diagnostics on the frozen test products; they were not used to select a model.

             mask               variable            level  n_cells  n_stations  mean_gain_mae    ci_low  ci_high  message_mae  null_mae  mean_abs_message_delta
       e2a_strict              age_group   never_observed       50           2       0.018092  0.000000 0.025846     0.561184  0.579276                0.023836
       e2a_strict              age_group         older_12     1752          60       0.054638  0.034725 0.076262     1.166852  1.221490                0.017286
       e2a_strict              age_group       recent_1_3       85          36       0.060823  0.027627 0.096020     0.946783  1.007606                0.020779
       e2a_strict              age_group    seasonal_4_12      336          57       0.064760  0.040010 0.090694     1.216543  1.281303                0.017870
       e2a_strict upstream_support_group             none     2223          69       0.055582  0.035984 0.077943     1.152325  1.207908                0.017655
       e2a_strict               flow_bin              0-1      505          25       0.049505  0.016798 0.084503     0.684525  0.734029                0.017715
       e2a_strict               flow_bin             1-10      150          14       0.033323  0.000000 0.101525     1.169883  1.203206                0.004884
       e2a_strict               flow_bin           10-100      331          19       0.023508 -0.000131 0.067763     1.894154  1.917662                0.005638
       e2a_strict               flow_bin           100-1k      388          26       0.035242 -0.002832 0.081075     1.589595  1.624837                0.009744
       e2a_strict               flow_bin         10k-100k      335          18       0.087627  0.059380 0.111961     0.870964  0.958591                0.030727
       e2a_strict               flow_bin           1k-10k      377          25       0.071475  0.038575 0.108139     1.090743  1.162218                0.023680
       e2a_strict               flow_bin            >100k      137           6       0.115364  0.024182 0.178662     0.684243  0.799606                0.034318
e3_spatial_seed42              age_group   never_observed     2531          43       0.017920  0.003438 0.034690     2.956946  2.974866                0.005074
e3_spatial_seed42 upstream_support_group             none     1510          42       0.005851  0.000514 0.012400     3.467638  3.473489                0.002295
e3_spatial_seed42 upstream_support_group visible_upstream     1021          15       0.035770  0.003924 0.062444     2.201661  2.237431                0.009185
e3_spatial_seed42               flow_bin              0-1      871          27       0.006552 -0.001252 0.018536     3.688735  3.695287                0.005959
e3_spatial_seed42               flow_bin             1-10      257          22      -0.001970 -0.008112 0.003105     4.033207  4.031237                0.001958
e3_spatial_seed42               flow_bin           10-100      512          28       0.019434 -0.001734 0.041944     2.641819  2.661253                0.003450
e3_spatial_seed42               flow_bin           100-1k      387          24       0.006700 -0.001861 0.016367     2.332687  2.339386                0.002718
e3_spatial_seed42               flow_bin         10k-100k       32           3       0.059759 -0.031799 0.074816     1.929014  1.988773                0.010358
e3_spatial_seed42               flow_bin           1k-10k      188          13       0.016545 -0.000291 0.027566     1.664317  1.680862                0.004360
e3_spatial_seed42               flow_bin            >100k      284           1       0.079543       NaN      NaN     2.128963  2.208506                0.011196

## Reading

The temporal mask has no visible-upstream support in the test query, so its upstream-support contrast is not identifiable.
The spatial mask provides the relevant upstream-support contrast. Intervals are station-clustered and should be read as diagnostic evidence, not a new primary endpoint.
