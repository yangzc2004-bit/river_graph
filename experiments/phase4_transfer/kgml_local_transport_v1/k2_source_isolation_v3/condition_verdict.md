# K2 message gain by observation and hydrologic context

Positive gain means the message-only model has lower absolute error than the zero-message null.
Bins are descriptive diagnostics on the frozen test products; they were not used to select a model.

             mask               variable            level  n_cells  n_stations  mean_gain_mae    ci_low  ci_high  message_mae  null_mae  mean_abs_message_delta
       e2a_strict              age_group   never_observed       50           2      -0.013659 -0.019513 0.000000     0.592935  0.579276                0.047698
       e2a_strict              age_group         older_12     1752          60       0.113141  0.067913 0.163621     1.108349  1.221490                0.043392
       e2a_strict              age_group       recent_1_3       85          36       0.127143  0.042794 0.215044     0.880463  1.007606                0.063095
       e2a_strict              age_group    seasonal_4_12      336          57       0.139248  0.080451 0.203495     1.142054  1.281303                0.048037
       e2a_strict upstream_support_group             none     2223          69       0.114770  0.069297 0.166794     1.093137  1.207908                0.044944
       e2a_strict               flow_bin              0-1      505          25       0.092843  0.022545 0.167805     0.641187  0.734029                0.043672
       e2a_strict               flow_bin             1-10      150          14       0.082604  0.000000 0.252077     1.120602  1.203206                0.012443
       e2a_strict               flow_bin           10-100      331          19       0.054274 -0.000920 0.161392     1.863388  1.917662                0.014440
       e2a_strict               flow_bin           100-1k      388          26       0.086530 -0.007860 0.197845     1.538307  1.624837                0.025863
       e2a_strict               flow_bin         10k-100k      335          18       0.172220  0.086670 0.247900     0.786371  0.958591                0.079501
       e2a_strict               flow_bin           1k-10k      377          25       0.151532  0.075173 0.236250     1.010686  1.162218                0.061684
       e2a_strict               flow_bin            >100k      137           6       0.215320 -0.010534 0.358407     0.584287  0.799606                0.082392
e3_spatial_seed42              age_group   never_observed     2531          43       0.016781  0.003475 0.031755     2.958085  2.974866                0.004832
e3_spatial_seed42 upstream_support_group             none     1510          42       0.006402  0.000506 0.014095     3.467087  3.473489                0.002427
e3_spatial_seed42 upstream_support_group visible_upstream     1021          15       0.032131  0.003989 0.056024     2.205300  2.237431                0.008388
e3_spatial_seed42               flow_bin              0-1      871          27       0.006335 -0.000927 0.017061     3.688952  3.695287                0.005592
e3_spatial_seed42               flow_bin             1-10      257          22      -0.001889 -0.007931 0.003044     4.033126  4.031237                0.001905
e3_spatial_seed42               flow_bin           10-100      512          28       0.016953 -0.001835 0.036301     2.644300  2.661253                0.003130
e3_spatial_seed42               flow_bin           100-1k      387          24       0.007908 -0.001899 0.019500     2.331478  2.339386                0.002977
e3_spatial_seed42               flow_bin         10k-100k       32           3       0.057320 -0.035632 0.071799     1.931453  1.988773                0.010198
e3_spatial_seed42               flow_bin           1k-10k      188          13       0.019235 -0.000157 0.032000     1.661627  1.680862                0.004991
e3_spatial_seed42               flow_bin            >100k      284           1       0.071299       NaN      NaN     2.137207  2.208506                0.010034

## Reading

The temporal mask has no visible-upstream support in the test query, so its upstream-support contrast is not identifiable.
The spatial mask provides the relevant upstream-support contrast. Intervals are station-clustered and should be read as diagnostic evidence, not a new primary endpoint.
