# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `mask`; objective: `direct`.

feature_set objective              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
       mask    direct     e1_r20_seed42 4514         0.493895        1.449039            1.490756  1.428633
       mask    direct       e2b_partial 1778         0.569003        1.057108            0.802976  0.818274
       mask    direct        e2a_strict 2223         0.615391        1.083995            0.976260  0.928839
       mask    direct e3_spatial_seed42 2531         0.476487        2.679058            2.954307  2.752407

## Pooled test

        system      mae
          gate 1.533124
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

feature_set objective              mask candidate       baseline  gain_mae     ci_lo    ci_hi
       mask    direct     e1_r20_seed42      gate     rf_context  0.035936 -0.007255 0.077671
       mask    direct     e1_r20_seed42      gate residual_nomsg  0.073900  0.027141 0.135491
       mask    direct       e2b_partial      gate     rf_context  0.243146  0.128084 0.381691
       mask    direct       e2b_partial      gate residual_nomsg -0.033082 -0.131119 0.041189
       mask    direct        e2a_strict      gate     rf_context  0.168079  0.014968 0.344230
       mask    direct        e2a_strict      gate residual_nomsg  0.014888 -0.094636 0.111736
       mask    direct e3_spatial_seed42      gate     rf_context -0.070392 -0.175001 0.030581
       mask    direct e3_spatial_seed42      gate residual_nomsg  0.223916  0.110210 0.333910
