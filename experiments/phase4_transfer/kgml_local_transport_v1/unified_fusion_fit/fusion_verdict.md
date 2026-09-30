# Trainable unified fusion pilot

The two expert predictions were frozen. A separate observable-feature gate was fitted for each family using validation labels only, then evaluated on the terminal test predictions.

             mask    n  feature_set  init_alpha  test_alpha_mean  test_alpha_sd  rf_context_mae  residual_nomsg_mae  fusion_mae
    e1_r20_seed42 4514 disagreement        0.40         0.442849       0.088195        1.449039            1.490756    1.426194
      e2b_partial 1778 disagreement        0.74         0.666264       0.110244        1.057108            0.802976    0.800390
       e2a_strict 2223 disagreement        0.74         0.680742       0.115775        1.083995            0.976260    0.896245
e3_spatial_seed42 2531 disagreement        0.00         0.009953       0.000872        2.679058            2.954307    2.678907

## Station-clustered effects

             mask       baseline  gain_mae     ci_lo    ci_hi
    e1_r20_seed42     rf_context  0.038366  0.010159 0.066473
    e1_r20_seed42 residual_nomsg  0.076330  0.013824 0.152838
      e2b_partial     rf_context  0.224665  0.146118 0.318721
      e2b_partial residual_nomsg -0.051563 -0.213424 0.060694
       e2a_strict     rf_context  0.171357  0.051919 0.290950
       e2a_strict residual_nomsg  0.018166 -0.170072 0.171316
e3_spatial_seed42     rf_context  0.000316 -0.001957 0.002485
e3_spatial_seed42 residual_nomsg  0.294623  0.092192 0.500770
